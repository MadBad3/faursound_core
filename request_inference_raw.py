import requests
import os
import progressbar

URL = 'http://localhost:8000/EOL/'

def inference_wav_raw(file_path, url = URL):
    files = {'wav': open(file_path, 'rb')}
    try:
        r = requests.post(url, files=files)
    except requests.exceptions.Timeout:
        r = requests.post(url, files=files)
        print(f'this is 2nd time retry to connect to API')
    except requests.exceptions.TooManyRedirects:
        print(f'The URL was bad and try a different one')
    except requests.exceptions.RequestException as e:
        # catastrophic error. bail.
        raise SystemExit(e)
    return r.json()

def process_one_wav_folder(wav_folder):
    file_lists = os.listdir(wav_folder)
    print(f'processing {len(file_lists)} wav files..')

    widgets = ['request to API running : ', progressbar.AnimatedMarker(), progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()] 
    bar = progressbar.ProgressBar(widgets=widgets, maxval=len(file_lists)).start()

    for count, file_name in enumerate(file_lists):
        file_path = os.path.join(wav_folder, file_name)
        
        response = inference_wav_raw(file_path)
        bar.update(count)
        print(f'{count} response from faursound-api = {response}')


if __name__ == "__main__":
    wav_folder = r'D:\Github\FaurSound\data\01 fev\Waves'
    all_sub_folders = [ os.path.join(wav_folder,name) for name in os.listdir(wav_folder) if os.path.isdir(os.path.join(wav_folder, name)) ]

    for folder in all_sub_folders:
        process_one_wav_folder(folder)
