import tensorflow as tf # import tensorflow
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("TkAgg") # necessary for plot
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
import cv2
from typing import List
import os
import sys
# sys.path.append((os.path.dirname(os.path.dirname(__file__))))
from _wavNVH import _wavNVH

# importing all scripts that will be needed to export your model and use it for inference
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

class inference(object):

    def __init__(self, path2label_map:str, detection_model:object) -> None:
        """init function for inference

        :param path2label_map: [description]
        :type path2label_map: str
        :param detection_model: [description]
        :type detection_model: object
        """

        self.category_index = label_map_util.create_category_index_from_labelmap(path2label_map,use_display_name=True)
        self.detection_model = detection_model
        

    def detect_fn(self,image):
        """
        Detect objects in image.
        
        Args:
        image: (tf.tensor): 4D input image (axis 0 is batch)
        
        Returs:
        detections (dict): predictions that model made
        """

        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)

        return detections


    def load_image_into_numpy_array(self, path):
        """Load an image from file into a numpy array.

        Puts image into numpy array to feed into tensorflow graph.
        Note that by convention we put it into a numpy array with shape
        (height, width, channels), where channels=3 for RGB.

        Args:
        path: the file path to the image

        Returns:
        numpy array with shape (img_height, img_width, 3)
        """
        
        return np.array(Image.open(path))


    ##run inference and plot results an an input image:
    def inference_with_plot(self,path2images, box_th=0.25):
        """
        Function that performs inference and plots resulting b-boxes
        
        Args:
        path2images: an array with pathes to images
        box_th: (float) value that defines threshold for model prediction.
        
        Returns:
        None
        """
        for image_path in path2images:

            print('Running inference for {}... '.format(image_path), end='')

            image_np = self.load_image_into_numpy_array(image_path)
            
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = self.detect_fn(input_tensor)

            # All outputs are batches tensors.
            # Convert to numpy arrays, and take index [0] to remove the batch dimension.
            # We're only interested in the first num_detections.
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    self.category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=200,
                    min_score_thresh=box_th,
                    agnostic_mode=False,
                    line_thickness=5)

            plt.figure(figsize=(15,10))
            plt.imshow(image_np_with_detections)
            print('Done')
        plt.show()


    ##other supporting functions:
    def nms(self, rects, thd=0.5):
        """
        Filter rectangles
        rects is array of oblects ([x1,y1,x2,y2], confidence, class)
        thd - intersection threshold (intersection divides min square of rectange)
        """
        out = []

        remove = [False] * len(rects)

        for i in range(0, len(rects) - 1):
            if remove[i]:
                continue
            inter = [0.0] * len(rects)
            for j in range(i, len(rects)):
                if remove[j]:
                    continue
                inter[j] = self.intersection(rects[i][0], rects[j][0]) / min(self.square(rects[i][0]), self.square(rects[j][0]))

            max_prob = 0.0
            max_idx = 0
            for k in range(i, len(rects)):
                if inter[k] >= thd:
                    if rects[k][1] > max_prob:
                        max_prob = rects[k][1]
                        max_idx = k

            for k in range(i, len(rects)):
                if (inter[k] >= thd) & (k != max_idx):
                    remove[k] = True

        for k in range(0, len(rects)):
            if not remove[k]:
                out.append(rects[k])

        boxes = [box[0] for box in out]
        scores = [score[1] for score in out]
        classes = [cls[2] for cls in out]
        return boxes, scores, classes


    def intersection(self, rect1, rect2):
        """
        Calculates square of intersection of two rectangles
        rect: list with coords of top-right and left-boom corners [x1,y1,x2,y2]
        return: square of intersection
        """
        x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]));
        y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]));
        overlapArea = x_overlap * y_overlap;
        return overlapArea


    def square(self, rect):
        """
        Calculates square of rectangle
        """
        return abs(rect[2] - rect[0]) * abs(rect[3] - rect[1])


    # *Next function is the one that you can use to run inference and save results into a file
    def inference_as_raw_output(self, image_path, box_th = 0.5, nms_th = 0.5, to_file = False):
        """
        Function that performs inference and return filtered predictions
        
        Args:
        image_path: image path
        box_th: (float) value that defines threshold for model prediction. Consider 0.5 as a value.
        nms_th: (float) value that defines threshold for non-maximum suppression. Consider 0.5 as a value.
        to_file: (boolean). When passed as True => results are saved into a file. Writing format is
        path2image + (x1abs, y1abs, x2abs, y2abs, score, conf) for box in boxes
        Returs:
        detections (dict): filtered predictions that model made
        """

        image_np = self.load_image_into_numpy_array(image_path)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)
        
        # checking how many detections we got
        num_detections = int(detections.pop('num_detections'))
        
        # filtering out detection in order to get only the one that are indeed detections
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        
        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        
        # defining what we need from the resulting detection dict that we got from model output
        key_of_interest = ['detection_classes', 'detection_boxes', 'detection_scores']
        
        # filtering out detection dict in order to get only boxes, classes and scores
        detections = {key: value for key, value in detections.items() if key in key_of_interest}
        
        if box_th: # filtering detection if a confidence threshold for boxes was given as a parameter
            for key in key_of_interest:
                scores = detections['detection_scores']
                current_array = detections[key]
                filtered_current_array = current_array[scores > box_th]
                detections[key] = filtered_current_array
        
        if nms_th: # filtering rectangles if nms threshold was passed in as a parameter
            # creating a zip object that will contain model output info as
            output_info = list(zip(detections['detection_boxes'],
                                detections['detection_scores'],
                                detections['detection_classes']
                                )
                            )
            boxes, scores, classes = self.nms(output_info,thd=nms_th)
            
            detections['detection_boxes'] = boxes # format: [y1, x1, y2, x2]
            detections['detection_scores'] = scores
            detections['detection_classes'] = classes
            
        if to_file: # if saving to txt file was requested

            image_h, image_w, _ = image_np.shape

            dir_path=os.path.join(os.path.dirname(image_path),'prediction_txt_raw_output')

            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)

            title , _ = os.path.splitext(os.path.basename(image_path))

            file_name = os.path.join(dir_path, f'{title}.txt')
            
            line2write = list()
            line2write.append(os.path.basename(image_path))
            
            with open(file_name, 'w') as text_file:
                # iterating over boxes
                for b, s, c in zip(boxes, scores, classes):
                    
                    y1abs, x1abs = b[0] * image_h, b[1] * image_w
                    y2abs, x2abs = b[2] * image_h, b[3] * image_w
                    
                    list2append = [x1abs, y1abs, x2abs, y2abs, s, c]
                    line2append = ','.join([str(item) for item in list2append])
                    
                    line2write.append(line2append)
                
                line2write = '\n'.join(line2write)
                text_file.write(line2write + os.linesep)
            print(f'save prediction result to {file_name} -> Done')
            
        return detections


    def inference_one_img_with_plot(self, img_path:str, min_score_thresh = 0.5) -> None :
        """inference_one_img_with plot

        :param img_path: [image path]
        :type img_path: [str]
        :param min_score_thresh: [min confident score to plot], defaults to 0.5
        :type min_score_thresh: float, optional
        """
        image_np = self.load_image_into_numpy_array(img_path)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    self.category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=10,
                    min_score_thresh=min_score_thresh,
                    agnostic_mode=False)

        plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
        fig = plt.gcf()
        fig.set_size_inches(15, 10)
        plt.show()


    def inference_one_img_with_plot_on_stft(self,cv_img_path:str,stft_img_path:str,
                                            save_png_folder:str = "" ,min_score_thresh = 0.5) -> None:
        """inference one image cv then plot on stft image

        :param cv_img_path: [cv picture path]
        :type cv_img_path: [str]
        :param stft_img_path: [stft picture path]
        :type stft_img_path: [str]
        :param save_png_folder: [save path for output png picture with detections], defaults to ""
        :type save_png_folder: str, optional
        :param min_score_thresh: [min confident score to plot], defaults to 0.5
        :type min_score_thresh: float, optional
        """
        image_np = self.load_image_into_numpy_array(cv_img_path)
        image_np_stft = self.load_image_into_numpy_array(stft_img_path)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np_stft.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    self.category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=10,
                    min_score_thresh=min_score_thresh,
                    agnostic_mode=False)

        fig, ax = plt.subplots(figsize=_wavNVH.FIGSIZE)

        ax.imshow(image_np_with_detections)
        ax.axis('off')

        if not save_png_folder:
            plt.show()

        else :
            if not os.path.isdir(save_png_folder):
                os.mkdir(save_png_folder)

            title , _ = os.path.splitext(os.path.basename(stft_img_path))

            file_name = os.path.join(save_png_folder, f'{title}.png')
            plt.savefig(file_name, bbox_inches='tight', pad_inches=0, transparent=False)
            
            print(f'save inference result picture for {file_name} -> DONE ')
            plt.close('all')


    def inference_folder_with_plot_on_stft(self,cv_folder:str,stft_folder:str,output_folder:str) -> None:
        """inference_one_img_with_plot_on_stft on full folder

        :param cv_folder: [folder path with cv pictures]
        :type cv_folder: str
        :param stft_folder: [folder path with stft pictures]
        :type stft_folder: str
        :param output_folder: [output folder for image with detections]
        :type output_folder: str
        """
        path_list_cv = self._get_jpg_file_list_from_folder(cv_folder)

        for cv_file in path_list_cv:
            fn = os.path.basename(cv_file)
            print(f'running prediction on {fn}')

            stft_file = os.path.join(stft_folder,fn)
            self.inference_one_img_with_plot_on_stft(cv_file,stft_file, save_png_folder = output_folder)


    def inference_lists_with_plot_on_stft(self,cv_pic_list:List,stft_pic_list:List,output_folder:str) -> None:
        """inference_one_img_with_plot_on_stft on list

        :param cv_pic_list: [cv picture list]
        :type cv_pic_list: List
        :param stft_pic_list: [stft picture list]
        :type stft_pic_list: List
        :param output_folder: [output folder path]
        :type output_folder: str
        """
        for idx in range(len(cv_pic_list)):
            self.inference_one_img_with_plot_on_stft(cv_pic_list[idx],stft_pic_list[idx], 
                                                    save_png_folder = output_folder)


    def inference_folder_as_raw_output(self,cv_folder:str) -> None:
        """inference whole folder as raw output

        :param cv_folder: [cv folder path]
        :type cv_folder: str
        """
        path_list_cv = self._get_jpg_file_list_from_folder(cv_folder)

        for cv_file in path_list_cv:
            print(f'running prediction on {cv_file} -> output txt file')

            self.inference_as_raw_output(cv_file, to_file = True)


    def _get_jpg_file_list_from_folder(self, folder_path:str) -> List:
        """get jpg file name list from a folder

        :param folder_path: [folder path]
        :type folder_path: str
        :return: [path list]
        :rtype: [list]
        """
        path_list = []
        for file in os.listdir(folder_path):
            _ , ext = os.path.splitext(file)
            if ext == '.jpg':
                path_list.append(os.path.join(folder_path,file))
        return path_list