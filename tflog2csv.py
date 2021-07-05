from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import pandas as pd
from PIL import Image
from tensorflow.python.framework import tensor_util
import tensorflow as tf
import glob
import re
import numpy as np

def tfevent2csv(input_folder: str, output_folder:str, output_folder_img:str, csv_file_name:str):
    def clean_up_df(df_orig):
        df_orig.rename(columns={"Values": "AP"}, inplace=True)

        df_ = df_orig.iloc[1:,:].copy()

        def get_label_name(row):
            string_to_check = row['Name']
            m = re.search(r'@0.5IOU/(\w*)', string_to_check)
            return m.group(1)
        df_['label'] = df_.apply(get_label_name, axis=1)
        df = df_[['label','AP']]

        df_output = df.append({'label':'mAP_without_1st','AP':round(df['AP'][1:].mean(),2)}, ignore_index=True)  #* when calculate mAP ignore the 1st label (due to issue from tf AP calculation)
        return df_output

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    if not os.path.isdir(output_folder_img):
        os.mkdir(output_folder_img)

    event_acc = EventAccumulator(input_folder)
    event_acc.Reload()
    # Show all tags in the log file
    tags = event_acc.Tags()['tensors']
    
    newTags = [name for name in tags if 'AP' in name]

    imgList = [name for name in event_acc.Tags()['tensors'] if 'eval_side_by_side' in name]

    values = []
    length = len(event_acc.Tensors('PascalBoxes_Precision/mAP@0.5IOU'))

    for i in range(length):
        for name in newTags:
            values.append(tensor_util.MakeNdarray(event_acc.Tensors(name)[i].tensor_proto).tolist())

        df = pd.DataFrame(data=newTags, columns=['Name'])
        df['Values'] = [round(num, 2) for num in values]
        
        values = []
        output_file = os.path.join(output_folder,csv_file_name)
        df_output = clean_up_df(df)
        df_output.to_csv(output_file, index=False)

        for name in imgList:
            img = tensor_util.MakeNdarray(event_acc.Tensors(name)[i].tensor_proto)
            tf_img = tf.image.decode_image(img[2])
            np_img = tf_img.numpy()
            im = Image.fromarray(np_img)
            im.save(os.path.join(output_folder_img,f'{str(i)}-{name}.png'))


def check_if_have_tfevent(folder_path:str) -> bool:
    """check if folder have tfevent

    :param folder_path: [folder path to check]
    :type folder_path: str
    :return: [have tfevent or not]
    :rtype: bool
    """
    file_list = glob.glob(folder_path+'/*events.out.tfevents*' , recursive=False)
    if file_list:
        return True
    else:
        return False


# def process_whole_folder(model_path):
#     all_sub_folders = [ name for name in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, name)) ]
#     model_versions = [x for x in all_sub_folders if check_if_have_tfevent(os.path.join(model_path, x))]

#     print('processing :', ', '.join(model_versions))
#     csv_folder = os.path.join(model_path, 'mAP_Results')
#     image_folder = os.path.join(model_path, 'eval_images')
#     if not os.path.isdir(image_folder):
#         os.mkdir(image_folder)
        
#     for model_version in model_versions:
#         input_folder = os.path.join(model_path, model_version)
#         output_folder_img = os.path.join(image_folder, f'{model_version}')
#         tfevent2csv(input_folder = input_folder, output_folder = csv_folder,
#                 output_folder_img = output_folder_img, csv_file_name= f'{model_version}.csv')


def process_eval_folder(model_eval_path, model_version):
    model_path = os.path.dirname(model_eval_path)
    if check_if_have_tfevent(model_eval_path):
        csv_folder = os.path.join(model_path, 'mAP_Results')
        image_folder = os.path.join(model_path, 'eval_images')
        if not os.path.isdir(csv_folder):
            os.mkdir(csv_folder)
        if not os.path.isdir(image_folder):
            os.mkdir(image_folder)
            
        output_folder_img = os.path.join(image_folder, f'{model_version}')
        tfevent2csv(input_folder = model_eval_path, output_folder = csv_folder,
                output_folder_img = output_folder_img, csv_file_name= f'{model_version}.csv')
        print(f'processing for {model_version} -> Done')
    else :
        print(f'input folder do not have tfevent files')


def create_release_info_csv(model_version:str, output_folder:str,mAP_result_folder:str, xml_count_folder:str):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)


    xml_train_count = os.path.join(xml_count_folder, f'{model_version}_train_class_numbers.txt')
    xml_test_count = os.path.join(xml_count_folder, f'{model_version}_test_class_numbers.txt')

    mAP_files = glob.glob(f'{mAP_result_folder}/*{model_version}*.csv')
    mAP_file = mAP_files[0]

    df_orig = pd.read_csv(mAP_file)

    df_train = pd.read_csv(xml_train_count, header = None, sep = ':', names = ['label','train_count']).astype({'train_count':'int32'})
    df_train['label'] = df_train['label'].str.strip()

    df_test = pd.read_csv(xml_test_count, header = None, sep = ':', names = ['label','test_count']).astype({'test_count':'int32'})
    df_test['label'] = df_test['label'].str.strip()

    df_count = df_train.merge(df_test, on='label')

    df = df_orig.merge(df_count, on='label',how = 'outer')
    df['train_count'] = df['train_count'].fillna(0).astype(np.int64)
    df['test_count'] = df['test_count'].fillna(0).astype(np.int64)

    df_output = pd.concat([df.iloc[-3:,:],df.iloc[:-3,:]],axis=0).reset_index(drop=True)

    output_fp = os.path.join(output_folder, f'{model_version}_release_info.csv')
    df_output.to_csv(output_fp, index=False)
    print(df_output)
    print(f'release info csv ready')

if __name__ == '__main__':
    os.chdir(r'D:\Github\FaurSound')
    model_version='v1-5-2'

    model_eval_path = f'Tensorflow\workspace\models\{model_version}_eval'

    mAP_result_folder = r'Tensorflow\workspace\models\mAP_Results'
    release_info_folder = r'Tensorflow\workspace\models\release_info'
    xml_count_folder = r'Tensorflow\workspace\images'
    process_eval_folder(model_eval_path=model_eval_path, model_version=model_version)
    create_release_info_csv(model_version=model_version, output_folder=release_info_folder,
                            mAP_result_folder=mAP_result_folder, xml_count_folder=xml_count_folder)


