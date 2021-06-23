import requests
import os
import progressbar

URL = 'http://192.168.1.47:5000/EOL'

def inference_wav(file_path, url = URL):
    files = {'wav': open(file_path, 'rb')}
    r = requests.post(url, files=files)


def process_one_wav_folder(wav_folder):
    file_lists = os.listdir(wav_folder)
    print(f'processing {len(file_lists)} wav files..')

    widgets = ['request to API running : ', progressbar.AnimatedMarker(), progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()] 
    bar = progressbar.ProgressBar(widgets=widgets, maxval=len(file_lists)).start()

    for count, file_name in enumerate(file_lists):
        file_path = os.path.join(wav_folder, file_name)
        
        inference_wav(file_path)
        bar.update(count)


if __name__ == "__main__":
    wav_folder = r'D:\Github\FaurSound\data\01 fev\Waves'
    all_sub_folders = [ name for name in os.listdir(wav_folder) if os.path.isdir(os.path.join(wav_folder, name)) ]

    for folder in all_sub_folders:
        process_one_wav_folder(folder)