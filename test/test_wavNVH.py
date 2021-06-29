import pytest, librosa, os, shutil, numpy as np
import sys
sys.path.append((os.path.dirname(os.path.dirname(__file__))))
from _wavNVH import _wavNVH
from mock import patch

test_folder_location = os.path.dirname(__file__)
filepath_wav = os.path.join(test_folder_location, r'E-Tilt_6202-PC1_2402749_20210201_052216_11ACF9023_750N - Up.wav')
filepath_mp3 = os.path.join(test_folder_location, r'E-Tilt_6202-PC1_2402749_20210201_052216_11ACF9023_750N - Up.mp3')

n_fft = 512
hop_length = 128
vmin = 0
vmax = 100
left_cut = 0.2
right_cut = 0.1
small_cut = 0.3
rounding = 2
multiplier = 1.5
sr = 22050

def test_check_wav_file_type():
    assert _wavNVH.check_wav_file_type(filepath_wav) == True
    assert _wavNVH.check_wav_file_type(filepath_mp3) == False

def test_get_fn_from_file_path():
    obj =  _wavNVH(filepath_wav, n_fft, hop_length)
    assert type(obj.fn) == str
    assert obj.fn == 'E-Tilt_6202-PC1_2402749_20210201_052216_11ACF9023_750N - Up.wav'

def test_load_audio():
    obj = _wavNVH(filepath_wav, n_fft, hop_length)
    assert type(obj.y) == np.ndarray
    assert not None in obj.y

def test_update_duration():
    obj = _wavNVH(filepath_wav, n_fft, hop_length)
    assert not obj.duration == None

def test_get_stft_log_spec():
    obj = _wavNVH(filepath_wav, n_fft, hop_length)
    assert type(obj.stft_spec) == np.ndarray
    assert not None in obj.stft_spec
    assert len(obj.stft_spec.shape) == 2

def test_cut_wav_in_second():
    obj = _wavNVH(filepath_wav, n_fft, hop_length)
    old_y = obj.y
    old_duration = obj.duration
    obj.cut_wav_in_second(left_cut, right_cut)
    assert round(old_duration-small_cut, rounding) == round(obj.duration, rounding) and old_y.shape[0]-sr*small_cut == obj.y.shape[0]

#never called
#def test_get_mel_spec():
#    obj = _wavNVH(filepath_wav, n_fft, hop_length)
#    assert type(obj.mel_spec) == np.ndarray
#    assert not None in obj.mel_spec

#@patch('matplotlib.pyplot.show')
#def test_plot_mel_spec(mock_show):
#    obj = _wavNVH(filepath_wav, n_fft, hop_length)
#    obj.plot_mel_spec(0, 100)
#    mock_show.assert_called_once()

@patch('matplotlib.pyplot.show')
def test_plot_stft_spec(mock_show2):
    obj = _wavNVH(filepath_wav, n_fft, hop_length)
    obj.plot_stft_spec(vmin, vmax)
    mock_show2.assert_called_once()

@patch('builtins.print')
def test_stft_spec_to_pic(mock_print):
    obj = _wavNVH(filepath_wav, n_fft, hop_length)
    obj.stft_spec_to_pic('test_folder', vmin, vmax)
    title, _ = os.path.splitext(obj.fn)
    path = os.path.join('test_folder', title + '.jpg')
    _ , file_extension = os.path.splitext(path)
    assert file_extension == '.jpg'
    os.remove(path)
    shutil.rmtree('./test_folder', ignore_errors=True)
    mock_print.assert_called_once()

#never called because so self.mel_spec == None
#@patch('matplotlib.pyplot.savefig')
#@patch('builtins.print')
#def test_mel_spec_to_pic(self, mock_savefig2, mock_os_mkdir2):
#    obj = _wavNVH(filepath_wav, n_fft, hop_length)
#    obj.mel_spec_to_pic('test_folder', vmin, vmax)
#    mock_savefig2.assert_called_once()
#    mock_print2.assert_called_once()
    
def test_plot_wave():
    obj = _wavNVH(filepath_wav, n_fft, hop_length)
    obj.plot_wave()

@patch('builtins.print')
def test_set_wav_length(mock_print2):
    obj = _wavNVH(filepath_wav, n_fft, hop_length)
    initial_duration = obj.duration
    with pytest.raises(KeyError):
        obj.set_wav_length(initial_duration*multiplier)
    obj.set_wav_length(initial_duration*small_cut)
    assert round(initial_duration*small_cut,rounding) == round(obj.duration,rounding)
    mock_print2.assert_called_once()

def test_coord_time():
    frames, sr = librosa.load(filepath_wav)
    _wavNVH._coord_time(frames.shape[0])

def test_coord_fft_hz():
    frames, sr = librosa.load(filepath_wav)
    _wavNVH._coord_fft_hz(frames.shape[0])
