import requests
import socket
import requests.packages.urllib3.util.connection as urllib3_cn


def allowed_gai_family():
    """
        https://github.com/shazow/urllib3/blob/master/urllib3/util/connection.py
    """
    return socket.AF_INET       #* this to force use ipv4 (issue with ipv6, get 2s delay)

urllib3_cn.allowed_gai_family = allowed_gai_family


URL = 'http://localhost:8000/EOL'
# URL = 'http://localhost:8000/test'
# URL_HELLO = 'http://localhost:8000/hello'


def labview_call_api(file_path, url = URL):
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