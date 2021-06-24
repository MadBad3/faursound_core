import requests
import os
import progressbar
import time
import json
import pprint

URL = 'http://localhost:8000/EOL'

def inference_wav_raw(file_path, url = URL):
    t1 = time.time()
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
    t2 = time.time()
    return (r.json(), (t2 - t1))


def process_one_wav_folder(wav_folder, log_file_path):
    file_lists = os.listdir(wav_folder)
    print(f'processing {len(file_lists)} wav files..')

    # widgets = ['request to API running : ', progressbar.AnimatedMarker(), progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()] 
    # bar = progressbar.ProgressBar(widgets=widgets, maxval=len(file_lists)).start()

    for count, file_name in enumerate(file_lists):
        file_path = os.path.join(wav_folder, file_name)
        
        response, time_spend = inference_wav_raw(file_path)
        # bar.update(count)
        pprint.pprint(response)
        update_log_file(log_file_path, str(round(time_spend,3)))


def update_log_file(log_file, infor:str):
    with open(log_file, "a") as f:
        f.write(infor) 
        f.write("\n")


if __name__ == "__main__":
    wav_folder = r'D:\Github\FaurSound\data\01 fev\Waves'
    log_file = r'./log/api_log.txt'
    all_sub_folders = [ os.path.join(wav_folder,name) for name in os.listdir(wav_folder) if os.path.isdir(os.path.join(wav_folder, name)) ]

    for folder in all_sub_folders:
        process_one_wav_folder(folder, log_file)
