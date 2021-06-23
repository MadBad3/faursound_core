import requests
import os

URL = 'http://192.168.1.47:5000/EOL'

def inference_wav(file_path, url = URL):
    files = {'wav': open(file_path, 'rb')}
    r = requests.post(url, files=files)


if __name__ == "__main__":
    wav_folder = r'D:\Github\FaurSound\data\01 fev\Waves\Down'
    for file_name in os.listdir(wav_folder):
        file_path = os.path.join(wav_folder, file_name)

        inference_wav(file_path)

