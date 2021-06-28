import requests
import os
import progressbar
import time
import json
import logging
import http.client
import cProfile
import pstats
from requests.sessions import session
import socket
import requests.packages.urllib3.util.connection as urllib3_cn

# http.client.HTTPConnection.debuglevel = 1
# logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)
# requests_log = logging.getLogger('requsets.packages.urllib3')
# requests_log.setLevel(logging.WARNING)
# requests_log.propagate = True

def allowed_gai_family():
    """
        https://github.com/shazow/urllib3/blob/master/urllib3/util/connection.py
    """
    return socket.AF_INET

urllib3_cn.allowed_gai_family = allowed_gai_family


URL = 'http://localhost:8000/EOL/'
# URL = 'http://localhost:8000/test/'
URL_HELLO = 'http://localhost:8000/hello/'

def inference_wav_raw(file_path,  profiling_fp:str, url = URL):
    with cProfile.Profile() as pr_:
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

    stats = pstats.Stats(pr_)
    stats.sort_stats(pstats.SortKey.TIME)

    stats.dump_stats(filename=profiling_fp)
    return (r.json(), (t2 - t1))


def process_one_wav_folder(wav_folder, log_file_path, profiling_folder:str):
    file_lists = os.listdir(wav_folder)
    print(f'processing {len(file_lists)} wav files..')

    # widgets = ['request to API running : ', progressbar.AnimatedMarker(), progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()] 
    # bar = progressbar.ProgressBar(widgets=widgets, maxval=len(file_lists)).start()

    for count, file_name in enumerate(file_lists):
        file_path = os.path.join(wav_folder, file_name)
        profiling_fp = os.path.join(profiling_folder, f'{count}.prof')
        response, time_spend = inference_wav_raw(file_path, profiling_fp)
        # bar.update(count)
        print(response)
        print(f'time spend = {time_spend}')
        update_log_file(log_file_path, str(round(time_spend,3)))


def update_log_file(log_file, infor:str):
    with open(log_file, "a") as f:
        f.write(infor) 
        f.write("\n")


def hello_request(url = URL_HELLO, session = False):
    t1 = time.time()
    if session:
        print('using requests.Session()')
        session_ = requests.Session()
        r = session_.get(url)
    else:
        r = requests.get(url)
    t2 = time.time()
    return (r.json(), (t2 - t1))


if __name__ == "__main__":
    # wav_folder = r'.\data\01 fev\Waves'
    wav_folder = r'D:\Github\FaurSound\data\01 fev\Waves'
    now = round(time.time(),2)
    log_file = f'./log/api_log_{now}.txt'
    all_sub_folders = [ os.path.join(wav_folder,name) for name in os.listdir(wav_folder) if os.path.isdir(os.path.join(wav_folder, name)) ]

    profiling_folder = r'./speed_profiling'
    if not os.path.isdir(profiling_folder):
        os.mkdir(profiling_folder)
    for folder in all_sub_folders:
        process_one_wav_folder(wav_folder = folder, log_file_path = log_file, profiling_folder = profiling_folder)


