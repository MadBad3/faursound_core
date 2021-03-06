import os
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import re
from configparser import ConfigParser
from datetime import datetime
import glob
import xml.etree.ElementTree as ET
from etiltEOL import etiltEOL
import sys
from typing import List
from xmlCounter import count_xml_and_prepare_tfrecord

class fsAzureStorage(object):

    def __init__(self, model_version:str='', for_training_sample = False, testing = False) -> None:
        #! model_version have to follow naming rule : 'v1-4-0'
        cfg = ConfigParser()
        self.for_training_sample = for_training_sample
        self.date = datetime.utcnow().strftime('%d%m%Y')
        self.model_version = model_version

        if not testing:
            cfg.read(os.path.join(os.path.dirname(__file__), 'secret','config.ini'))
            if self.for_training_sample:
                AZURE_KEY = cfg['DEFAULT']['AZURE_KEY_TRAINING']
                self.training_data_container = f'etilt-cv-training'

            else : 
                AZURE_KEY = cfg['DEFAULT']['AZURE_KEY_PREDICTION']
                self.predict_pic_container = f'{self.date}-predicts-{self.model_version}'
                self.txt_container = f'{self.date}-txt-{self.model_version}'
                self.cv_pic_container = f'{self.date}-cvpic'

            os.environ['AZURE_STORAGE_CONNECTION_STRING'] = AZURE_KEY
            connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            self.blob_service_client = BlobServiceClient.from_connection_string(connect_str)



    def _update_date(self):
        if self.for_training_sample:
            self.date = datetime.utcnow().strftime('%d%m%Y')
        else:
            self.date = datetime.utcnow().strftime('%d%m%Y')
            self.predict_pic_container = f'{self.date}-predicts-{self.model_version}'
            self.txt_container = f'{self.date}-txt-{self.model_version}'
            self.cv_pic_container = f'{self.date}-cvpic'


    def _creat_or_get_container_client(self,container_name):
        #* try to create the container, if failed , means it exist, then get_container_client
        try:
            container_client = self.blob_service_client.create_container(container_name)
        except :
            container_client = self.blob_service_client.get_container_client(container_name)
        return container_client


    def _upload_file(self,container_name, file_path, metadata = None):

        container_client = self._creat_or_get_container_client(container_name)

        with open(file_path, "rb") as data:
            blob_client = container_client.upload_blob(name=os.path.basename(file_path), data=data, 
                                                        metadata = metadata, overwrite = True)


    def _get_image_fp_based_on_xml_fp(self, xml_fp:str) -> str:
        """get image file path based on xml file name. they are same name, only difference is the extension

        :param xml_fp: [xml file path]
        :type xml_fp: str
        :return: [image file path]
        :rtype: str
        """
        base_name = os.path.basename(xml_fp)
        dirname = os.path.dirname(xml_fp)
        name, _ = os.path.splitext(base_name)
        img_file_name = name + '.jpg'
        img_file_path = os.path.join(dirname,img_file_name)
        return img_file_path


    def upload_prediction_to_azure(self,cv_pic_path, txt_path, predict_pic_path):
        #? to think about how to update metadata !
        metadata = {
            'labeled': 'no',
            'validated_label': 'no',
            'predicted': 'yes',
            'date':self.date,
            'location_checked': 'no',
            'location_corrected':'no',
            # 'current':'1.2',
        }
        self._update_date()
        self._upload_file(container_name = self.cv_pic_container, file_path=cv_pic_path, metadata = metadata)
        self._upload_file(container_name = self.txt_container, file_path=txt_path, metadata = metadata)
        self._upload_file(container_name = self.predict_pic_container, file_path=predict_pic_path, metadata = metadata)
        print(f'upload files to Azure blob storage -> Done ')


    def commit_training_sample(self, folder_to_upload):
        def get_direction_from_fn(file_name):
            #? do we need this to get direction automaticly?
            direction_ = re.search('Up|Down', file_name)
            if direction_:
                return direction_.group()
            else :
                return ""

        self._update_date()

        xml_file_list = glob.glob(folder_to_upload + '\*.xml')
        print(f'there are {len(xml_file_list)} files to upload')

        for xml_fp in xml_file_list:
            direction = get_direction_from_fn(os.path.basename(xml_fp))
            labels = self._get_noise_list_from_xml(xml_fp)
            metadata = {
            'validated_label': 'no',
            'direction': direction,
            'date':self.date,
            'labels': ','.join(labels) ,
            }
            cv_pic_fp = self._get_image_fp_based_on_xml_fp(xml_fp)
            self._upload_file(container_name = self.training_data_container, file_path = cv_pic_fp, metadata = metadata)
            self._upload_file(container_name = self.training_data_container, file_path = xml_fp, metadata = metadata)
            print(f'uploading files for {xml_fp} -> Done')
        print(f'commit training samples to Azure blob storage -> Done ')


    def pull_all_training_sample(self, local_folder, train_test_split=False, test_ratio:float = 0.1):
        if not os.path.isdir(local_folder):
            os.mkdir(local_folder)
        container_client = self.blob_service_client.get_container_client(self.training_data_container)
        blob_list = container_client.list_blobs()
        for blob in blob_list:
            blob_client = container_client.get_blob_client(blob)
            download_file_path = os.path.join(local_folder, blob.name)
            print("\nDownloading blob to \n\t" + download_file_path)

            with open(download_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
        if train_test_split:
            partition_dataset_script_path = os.path.join(os.path.dirname(__file__),'partition_dataset.py')
            os.system(f'python {partition_dataset_script_path} -i {local_folder} -r {test_ratio} -o {training_sample_dict} -x')


    def pull_training_sample_based_on_label(self, local_folder, labels:List):
        for label in labels:
            sub_folder = os.path.join(local_folder, label)
            if not os.path.isdir(sub_folder):
                os.mkdir(sub_folder)

        container_client = self.blob_service_client.get_container_client(self.training_data_container)
        blob_list = container_client.list_blobs(include='metadata')
        for blob in blob_list:
            metadata_labels = blob.metadata['labels']
            if metadata_labels:         # avoid if metadata_labels is empty
                blob_labels = metadata_labels.split(",")
                for blob_label in blob_labels:
                    if blob_label in labels:
                        blob_client = container_client.get_blob_client(blob)
                        download_file_path = os.path.join(local_folder, blob_label, blob.name)
                        print("\nDownloading blob to \n\t" + download_file_path)

                        with open(download_file_path, "wb") as download_file:
                            download_file.write(blob_client.download_blob().readall())
        print(f'pulling training sample based on label --> done')


    def update_blob_meta(self, container_name, blob_name, meta):
        pass


    def _get_noise_list_from_xml(self, xml_file):
        def count_class(class_name, file):
            number = total_class_number.get(class_name, None)
            if number is None:
                print('[Warning] Not found class name: %s in %s' % (class_name, file))
            else:
                total_class_number[class_name] = number + 1

        total_class_number = { key: 0 for key in etiltEOL.LABELS}
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except Exception:
            print("[Error] Cannot parse file %s" % xml_file)
            sys.exit(1)
        object_list = root.findall('object')
        if len(object_list) > 0:
            for target in object_list:
                count_class(target.find('name').text, xml_file)
        return [ key for key in total_class_number.keys() if total_class_number[key] > 0]



if __name__ == '__main__':
    os.chdir(r'D:\Github\FaurSound')

    azureClient = fsAzureStorage(model_version = '1-5-3', for_training_sample = True)

    training_sample_dict = r'D:\Github\FaurSound\Tensorflow\workspace\images'
    local_folder = os.path.join(training_sample_dict,azureClient.model_version)
    # azureClient.pull_training_sample_based_on_label(training_sample_dict, labels=['rattle_product'])
    # azureClient.commit_training_sample(folder_to_upload = r'D:\Github\FaurSound\Tensorflow\workspace\images\bench')
    azureClient.pull_all_training_sample(local_folder = local_folder, train_test_split=True)
    count_xml_and_prepare_tfrecord(model_version = azureClient.model_version, creat_tfrecord= True)







