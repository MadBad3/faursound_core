import tensorflow as tf # import tensorflow
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
import json

import viz_utils
from etiltEOL import etiltEOL
import base64


class inferenceEOL(object):

    def __init__(self, path2label_map_json:str,model_path:str,
                min_score_thresh = 0.5,nms_th = 0.5, ) -> None:

        self.category_index = self.load_label_map_from_json(path2label_map_json)
        self.detect_fn = tf.saved_model.load(model_path)
        self.min_score_thresh = min_score_thresh
        self.nms_th = nms_th
    

    def load_label_map_from_json(self,json_fn):
        def int_keys(ordered_pairs):
            result = {}
            for key, value in ordered_pairs:
                try:
                    key = int(key)
                except ValueError:
                    pass
                result[key] = value
            return result

        with open(json_fn) as f:
            data = json.load(f,object_pairs_hook=int_keys)
        return data


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


    def _get_json_input(self,image_path):
        with open(image_path, "rb") as image_file:
            jpeg_bytes = base64.b64encode(image_file.read()).decode('utf-8')
            request_body = {"instances": [{"b64": jpeg_bytes}]}
        return request_body

    #* this is the function modified for EOL
    def get_spec_pics_for_wav(self,wav_file, fn, output_folder: str) -> None:
        
        obj = etiltEOL(wav_file, n_fft = 512, hop_length=128,fn=fn)
        
        obj.cut_wav_in_second(left_cut=0.75, right_cut=0.75)

        cv_output_folder = os.path.join(output_folder,'cv')
        stft_output_folder = os.path.join(output_folder,'stft')
        obj.stft_custom_spec_to_pic(output_folder=stft_output_folder)
        obj.save_stft_cv_pic(output_folder=cv_output_folder)


    ##other supporting functions:
    def nms(self, rects):
        """
        Filter rectangles
        rects is array of oblects ([x1,y1,x2,y2], confidence, class)
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
                if inter[k] >= self.nms_th:
                    if rects[k][1] > max_prob:
                        max_prob = rects[k][1]
                        max_idx = k

            for k in range(i, len(rects)):
                if (inter[k] >= self.nms_th) & (k != max_idx):
                    remove[k] = True

        for k in range(0, len(rects)):
            if not remove[k]:
                out.append(rects[k])

        boxes = [box[0].tolist() for box in out]
        scores = [score[1].tolist() for score in out]
        classes = [cls[2].tolist() for cls in out]
        return np.asarray(boxes), np.asarray(scores), np.asarray(classes)


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


    def process_detections(self, detections):
        detections = {key: np.asarray(value) for key, value in detections.items()} # convert list to array for whole dict

        num_detections = int(detections.pop('num_detections'))
        
        # filtering out detection in order to get only the one that are indeed detections
        detections = {key: value[:num_detections] for key, value in detections.items()}
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        
        key_of_interest = ['detection_classes', 'detection_boxes', 'detection_scores']
        detections = {key: value for key, value in detections.items() if key in key_of_interest}
        
        if self.min_score_thresh: # filtering detection if a confidence threshold for boxes was given as a parameter
            for key in key_of_interest:
                scores = detections['detection_scores']
                current_array = detections[key]
                filtered_current_array = current_array[scores > self.min_score_thresh]
                detections[key] = filtered_current_array
        
        if self.nms_th: # filtering rectangles if nms threshold was passed in as a parameter
            # creating a zip object that will contain model output info as
            output_info = list(zip(detections['detection_boxes'],
                                detections['detection_scores'],
                                detections['detection_classes']
                                )
                            )
            boxes, scores, classes = self.nms(output_info)
            
            detections['detection_boxes'] = boxes # format: [y1, x1, y2, x2]
            detections['detection_scores'] = scores
            detections['detection_classes'] = classes
        return detections


    # *Next function is the one that you can use to run inference and save results into a file
    def inference_as_raw_output(self, image_path, to_file = False):
        """
        Function that performs inference and return filtered predictions
        
        Args:
        image_path: image path
        to_file: (boolean). When passed as True => results are saved into a file. Writing format is
        path2image + (x1abs, y1abs, x2abs, y2abs, score, conf) for box in boxes
        Returs:
        detections (dict): filtered predictions that model made
        """

        image_np = self.load_image_into_numpy_array(image_path)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)

        detections_raw = self.detect_fn(input_tensor)
        # checking how many detections we got
        detections = self.process_detections(detections_raw)

        detections_api = {key: value.tolist() for key, value in detections.items()} # convert array to list for whole dict for API

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
                for b, s, c in zip(
                                    detections['detection_boxes'], 
                                    detections['detection_scores'], 
                                    detections['detection_classes']
                                    ):
                    
                    y1abs, x1abs = b[0] * image_h, b[1] * image_w
                    y2abs, x2abs = b[2] * image_h, b[3] * image_w
                    
                    list2append = [x1abs, y1abs, x2abs, y2abs, s, c]
                    line2append = ','.join([str(item) for item in list2append])
                    
                    line2write.append(line2append)
                
                line2write = '\n'.join(line2write)
                text_file.write(line2write + os.linesep)
            # print(f'save prediction result to {file_name} -> Done')
            
        return detections_api


    def inference_one_img_with_plot_on_stft(self,cv_img_path,stft_img_path,save_png_folder:str = "",save_txt = False):
        
        image_np_cv = self.load_image_into_numpy_array(cv_img_path)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np_cv, 0), dtype=tf.uint8)

        detections_raw = self.detect_fn(input_tensor)
        detections = self.process_detections(detections_raw)

        if save_txt: # if saving to txt file was requested
            image_h, image_w, _ = image_np_cv.shape
            dir_path=os.path.join(os.path.dirname(cv_img_path),'prediction_txt_raw_output')

            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)

            title , _ = os.path.splitext(os.path.basename(cv_img_path))

            file_name = os.path.join(dir_path, f'{title}.txt')
            
            line2write = list()
            line2write.append(os.path.basename(cv_img_path))
            
            with open(file_name, 'w') as text_file:
                # iterating over boxes
                for b, s, c in zip(
                                    detections['detection_boxes'], 
                                    detections['detection_scores'], 
                                    detections['detection_classes']
                                    ):
                    
                    y1abs, x1abs = b[0] * image_h, b[1] * image_w
                    y2abs, x2abs = b[2] * image_h, b[3] * image_w
                    
                    list2append = [x1abs, y1abs, x2abs, y2abs, s, c]
                    line2append = ','.join([str(item) for item in list2append])
                    
                    line2write.append(line2append)
                
                line2write = '\n'.join(line2write)
                text_file.write(line2write + os.linesep)
            # print(f'save prediction result to {file_name} -> Done')

        image_np_stft = self.load_image_into_numpy_array(stft_img_path)
        image_np_with_detections = image_np_stft.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes'],
                    detections['detection_scores'],
                    self.category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=10,
                    min_score_thresh=self.min_score_thresh,
                    agnostic_mode=False)

        fig, ax = plt.subplots(figsize=(12.5,10))

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
            
            # print(f'save inference result picture for {file_name} -> DONE ')
            plt.close('all')

        return image_np_with_detections
