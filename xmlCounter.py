# -*- coding:utf-8 -*-
import os

import sys
sys.path.insert(0, r'./faursound_core')

from etiltWav import etiltWav

import xml.etree.ElementTree as ET


class xmlCounter(object):
    def __init__(self, input_folder:str, count_output_fp:str, label_list = etiltWav.LABELS):
        self.total_class_number = { key: 0 for key in label_list}

        self.input_folder = input_folder
        self.count_output_fp = count_output_fp
        self._get_xml_fp_list()
        self._read_xml_files()
        self._write_result()
        print('Counting finished!')


    def _get_xml_fp_list(self):
        xml_file_path_list = []
        for root, dirs, files in os.walk(self.input_folder):
            for file in files:
                if os.path.splitext(file)[1] == '.xml':
                    xml_file_path_list.append(os.path.join(root, file))
        self.xml_fp_list = xml_file_path_list


    def _count_class(self, class_name, file):
        number = self.total_class_number.get(class_name, None)
        if number is None:
            print('[Warning] Not found class name: %s in %s' % (class_name, file))
        else:
            self.total_class_number[class_name] = number + 1


    def _read_xml_files(self):
        for file in self.xml_fp_list:
            try:
                tree = ET.parse(file)
                root = tree.getroot()  # 获得根节点
            except Exception:
                print("[Error] Cannot parse file %s" % file)
                sys.exit(1)
            object_list = root.findall('object')
            if len(object_list) == 0:
                continue
            for target in object_list:
                self._count_class(target.find('name').text, file)


    def _write_result(self):
        count = 0
        with open(self.count_output_fp, 'w') as f:
            for key in self.total_class_number:
                f.write(key + ' : ' + str(self.total_class_number[key]) + '\r')
                count += self.total_class_number[key]
            f.write('Total numbers : ' + str(count) + '\r')
            f.write('Total files : ' + str(len(self.xml_fp_list)) + '\r')


def count_xml_and_prepare_tfrecord(model_version:str, creat_tfrecord=False):
    TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
    LABEL_MAP_NAME = 'label_map.pbtxt'
    LABEL_MAP_JSON = 'label_map.json'
    paths = {
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    }
    files = {
        'TF_RECORD_SCRIPT': os.path.join(os.path.dirname(__file__), TF_RECORD_SCRIPT_NAME), 
        'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME),
        'LABELMAP_JSON': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_JSON),
    }

    folder_list = [os.path.join(paths['IMAGE_PATH'],folder) for folder in ['train','test']]
    for xml_dir in folder_list:
        txt_fn = f'{model_version}_{os.path.basename(xml_dir)}_class_numbers.txt'
        result_file = os.path.join(os.path.dirname(xml_dir),txt_fn)
        xmlCounter(input_folder=xml_dir, count_output_fp=result_file)

    if creat_tfrecord:
        os.system(f"python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'train')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'train.record')}")
        os.system(f"python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'test')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'test.record')}")


if __name__ == '__main__':
    # path to annotation folder
    os.chdir(r'D:\Github\FaurSound')
    count_xml_and_prepare_tfrecord(model_version = "v1-5-1", creat_tfrecord= True)

