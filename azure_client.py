import os
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import re
from configparser import ConfigParser
from datetime import datetime
import glob

class fsAzureStorage(object):

    def __init__(self, model_version:str='', for_training_sample = False, testing = False) -> None:
        #! model_version have to follow naming rule : 'v1-4-0'
        cfg = ConfigParser()
        self.for_training_sample = for_training_sample
        self.date = datetime.utcnow().strftime('%d%m%Y')
        if not testing:
            cfg.read(r'./secret/config.ini')
            if self.for_training_sample:
                AZURE_KEY = cfg['DEFAULT']['AZURE_KEY_TRAINING']
                self.cv_pic_container = f'etilt-cv-training'

            else : 
                AZURE_KEY = cfg['DEFAULT']['AZURE_KEY_PREDICTION']
                self.model_version = model_version
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


    def commit_training_sample(self, folder_to_upload, direction:str):
        def _get_direction_from_fn(file_name):
            #? do we need this to get direction automaticly?
            direction_ = re.search('Up|Down', file_name)
            return direction_.group()

        self._update_date()

        metadata = {
            'validated_label': 'no',
            'direction': direction,
            'date':self.date,
        }
        xml_file_list = glob.glob(folder_to_upload + '\*.xml')

        for xml_fp in xml_file_list:
            cv_pic_fp = self._get_image_fp_based_on_xml_fp(xml_fp)
            self._upload_file(container_name = self.cv_pic_container, file_path = cv_pic_fp, metadata = metadata)
            self._upload_file(container_name = self.cv_pic_container, file_path = xml_fp, metadata = metadata)
            print(f'uploading files for {xml_fp} -> Done')
        print(f'commit training samples to Azure blob storage -> Done ')


    def pull_training_sample(self, local_folder):
        if not os.path.isdir(local_folder):
            os.mkdir(local_folder)
        container_client = self.blob_service_client.get_container_client(self.cv_pic_container)
        blob_list = container_client.list_blobs()
        for blob in blob_list:
            blob_client = container_client.get_blob_client(blob)
            download_file_path = os.path.join(local_folder, blob.name)
            print("\nDownloading blob to \n\t" + download_file_path)

            with open(download_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())


    def list_blobs_in_container(self, container_name):
        #TODO , to think about the use of this function
        container_client = self.blob_service_client.get_container_client(container_name)
        print('f')
        blob_list = container_client.list_blobs()
        for blob in blob_list:
            print(blob.name)


    def update_blob_meta(self, container_name, blob_name, meta):
        pass


if __name__ == '__main__':

    azureClient = fsAzureStorage(model_version = '1-3-0', for_training_sample = True)

    training_sample_dict = r'D:\Github\FaurSound\Tensorflow\workspace\images'
    sub_folder = r'v1-5-0'
    local_folder = os.path.join(training_sample_dict,sub_folder)

    azureClient.pull_training_sample(local_folder = local_folder)



