import os,uuid
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

from configparser import ConfigParser
from datetime import datetime


class fsAzureStorage(object):

    def __init__(self, model_version:str, testing = False) -> None:
        #! model_version have to follow naming rule : 'v1-4-0'
        cfg = ConfigParser()
        if not testing:
            cfg.read('config.ini')
            AZURE_KEY = cfg['DEFAULT']['AZURE_KEY']
            os.environ['AZURE_STORAGE_CONNECTION_STRING'] = AZURE_KEY
            connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            self.blob_service_client = BlobServiceClient.from_connection_string(connect_str)

        self.model_version = model_version
        self.date = datetime.utcnow().strftime('%d%m%Y')
        self.predict_pic_container = f'{self.date}-predicts-{self.model_version}'
        self.txt_container = f'{self.date}-txt-{self.model_version}'
        self.cv_pic_container = f'{self.date}-cvpic'


    def _update_date(self):
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


    def upload_to_azure(self,cv_pic_path, txt_path, predict_pic_path):
        #? to think about how to update metadata !
        metadata = {
            'labeled': 'no',
            'validated': 'no',
            'predicted': 'yes',
            'date':self.date,
        }
        self._update_date()
        self._upload_file(container_name = self.cv_pic_container, file_path=cv_pic_path, metadata = metadata)
        self._upload_file(container_name = self.txt_container, file_path=txt_path, metadata = metadata)
        self._upload_file(container_name = self.predict_pic_container, file_path=predict_pic_path, metadata = metadata)


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

    azureClient = fsAzureStorage(model_version = '1-3-0')


