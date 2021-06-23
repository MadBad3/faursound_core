import requests
import os
import progressbar

URL = 'http://192.168.1.47:5000/EOL'

def inference_wav(file_path, url = URL):
    files = {'wav': open(file_path, 'rb')}
    r = requests.post(url, files=files)


if __name__ == "__main__":
    wav_folder = r'D:\Github\FaurSound\data\01 fev\Waves\Down'

    file_lists = os.listdir(wav_folder)
    print(f'processing {len(file_lists)} wav files..')
    
    widgets = ['Spike detector running : ', progressbar.AnimatedMarker(), progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()] 
    bar = progressbar.ProgressBar(widgets=widgets, maxval=len(file_lists)).start()

    for count, file_name in enumerate(file_lists):
        file_path = os.path.join(wav_folder, file_name)
        
        inference_wav(file_path)
        bar.update(count)
