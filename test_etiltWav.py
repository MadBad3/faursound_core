import pytest, os, pandas as pd, numpy as np, cv2 as cv
import sys
# sys.path.append((os.path.dirname(os.path.dirname(__file__))))
from etiltWav import etiltWav
from mock import patch, call

n_fft = 512
hop_length = 128
output_folder = r'./test'


def test_load_df_meta():
	example = {'Type': ['example1','example2'],
        	'Price': [22000,25000]}

	df = pd.DataFrame(example)
	etiltWav.load_df_meta(df)

def test_df_meta():
	assert type(etiltWav.df_meta) == property

def test_label():
	assert type(etiltWav.label) == property

#def test_get_lables_based_on_fn():
#	pass

def test_get_info_from_filename():
	test_wav_up = r'./test/E-Tilt_6202-PC1_2402749_20210201_052216_11ACF9023_750N - Up.wav'
	obj = etiltWav(test_wav_up, n_fft=n_fft, hop_length=hop_length,get_infor_from_fn = True)
	assert obj.direction == 'Up'

	test_wav_down = r'./test/E-Tilt_6202-PC1_2402749_20210201_052216_11ACF9023_750N - Down.wav'
	obj = etiltWav(test_wav_down, n_fft=n_fft, hop_length=hop_length,get_infor_from_fn = True)
	assert obj.direction == 'Down'

	test_wav_default = r'./test/E-Tilt_6202-PC1_2402749_20210201_052216_11ACF9023_750N - Up.wav'
	obj = etiltWav(test_wav_default, n_fft=n_fft, hop_length=hop_length,get_infor_from_fn = False)
	assert obj.direction == None


@patch('matplotlib.pyplot.savefig')
@patch('builtins.print')
def test_stft_custom_spec_to_pic(mock_savefig, mock_print):
	test_wav = r'./test/E-Tilt_6202-PC1_2402749_20210201_052216_11ACF9023_750N - Down.wav'
	obj = etiltWav(test_wav, n_fft=n_fft, hop_length=hop_length,get_infor_from_fn = True)
	assert obj.spec_pic_path == None
	obj.stft_custom_spec_to_pic(output_folder=output_folder)
	mock_savefig.assert_called_once()
	mock_print.assert_called_once()
	assert type(obj.spec_pic_path) == str

@patch('matplotlib.pyplot.show')
def test_plot_stft_custom_spec(mock_show):
	test_wav = r'./test/E-Tilt_6202-PC1_2402749_20210201_052216_11ACF9023_750N - Down.wav'
	obj = etiltWav(test_wav, n_fft=n_fft, hop_length=hop_length,get_infor_from_fn = True)
	obj.plot_stft_custom_spec()
	mock_show.assert_called_once()

@patch('builtins.print')
def test_save_stft_cv_pic(mock_print2):
	test_wav = r'./test/E-Tilt_6202-PC1_2402749_20210201_052216_11ACF9023_750N - Down.wav'
	obj = etiltWav(test_wav, n_fft=n_fft, hop_length=hop_length,get_infor_from_fn = True)
	obj.save_stft_cv_pic(output_folder=output_folder)#with print
	assert mock_print2.mock_calls == [call('please save stft spec picture first !!')]
	
	assert obj.spec_pic_path == None
	obj.stft_custom_spec_to_pic(output_folder=output_folder)
	assert type(obj.spec_pic_path) == str

	obj.save_stft_cv_pic(output_folder=output_folder)

	assert type(obj.cv_prepocess(obj.spec_pic_path)) == np.ndarray
	assert mock_print2.mock_calls[2] == call('save cv picture for E-Tilt_6202-PC1_2402749_20210201_052216_11ACF9023_750N - Down.jpg -> DONE ')
	path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052216_11ACF9023_750N - Down.jpg')
	os.remove(path)

def test_cv_prepocess():
	test_wav = r'./test/E-Tilt_6202-PC1_2402749_20210201_052216_11ACF9023_750N - Down.wav'
	obj = etiltWav(test_wav, n_fft=n_fft, hop_length=hop_length,get_infor_from_fn = True)
	obj.stft_custom_spec_to_pic(output_folder=output_folder)
	color_img = obj.cv_prepocess(obj.spec_pic_path)
	assert type(color_img) == np.ndarray and len(color_img.shape) == 3
	path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052216_11ACF9023_750N - Down.jpg')
	os.remove(path)

	test_wav = r'./test/E-Tilt_6202-PC1_2402749_20210201_052216_11ACF9023_750N - Up.wav'
	obj = etiltWav(test_wav, n_fft=n_fft, hop_length=hop_length,get_infor_from_fn = True)
	obj.stft_custom_spec_to_pic(output_folder=output_folder)
	color_img = obj.cv_prepocess(obj.spec_pic_path)
	assert type(color_img) == np.ndarray and len(color_img.shape) == 3
	path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052216_11ACF9023_750N - Up.jpg')
	os.remove(path)

	test_wav = r'./test/E-Tilt_6202-PC1_2402749_20210201_052216_11ACF9023_750N - Right.wav'
	obj = etiltWav(test_wav, n_fft=n_fft, hop_length=hop_length,get_infor_from_fn = False)
	obj.stft_custom_spec_to_pic(output_folder=output_folder)
	obj.direction = 'Right'
	with pytest.raises(KeyError, match=r'etiltWav have direction not up and down -> can not do cv_preprocess'):
		obj.cv_prepocess(obj.spec_pic_path)
	path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052216_11ACF9023_750N - Right.jpg')
	os.remove(path)

def test_cv_prepocess_down():
	path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	test_wav = r'./test/E-Tilt_6202-PC1_2402749_20210201_052216_11ACF9023_750N - Down.wav'
	obj = etiltWav(test_wav, n_fft=n_fft, hop_length=hop_length,get_infor_from_fn = True)
	color_img = obj._cv_prepocess_down(path)
	assert type(color_img) == np.ndarray and len(color_img.shape) == 3
	assert not None in color_img

def test_cv_prepocess_up():
	path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	test_wav = r'./test/E-Tilt_6202-PC1_2402749_20210201_052216_11ACF9023_750N - Down.wav'
	obj = etiltWav(test_wav, n_fft=n_fft, hop_length=hop_length,get_infor_from_fn = True)
	color_img = obj._cv_prepocess_up(path)
	assert type(color_img) == np.ndarray and len(color_img.shape) == 3
	assert not None in color_img

def test_over_run_preprocessing():
	path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	test_wav = r'./test/E-Tilt_6202-PC1_2402749_20210201_052216_11ACF9023_750N - Down.wav'
	obj = etiltWav(test_wav, n_fft=n_fft, hop_length=hop_length,get_infor_from_fn = True)
	
	image = cv.imread(path)
	gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	img = obj.over_run_preprocessing(gray)

	assert type(img) == np.ndarray and len(img.shape) == 2
	assert not None in img