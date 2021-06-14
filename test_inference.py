import pytest, os, shutil, numpy as np
import sys
sys.path.append((os.path.dirname(os.path.dirname(__file__))))
from faursound_core.inference import inference
from mock import patch
from faursound_core.etiltWav import etiltWav

path = r'./test/folder'
path2label_map_json = r'./test/label_map.json'
path2img = r'./test/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg'
path2wav = r'./test/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.wav'
totalLabels = len(etiltWav.LABELS)
image_shape = (770, 968, 3)
x1 = 1
y1 = 250
x2 = 50
y2 = 300

@patch('tensorflow.saved_model.load')
def test_load_label_map_from_json(mock_tf_load):
	obj = inference(path2label_map_json,'model_path:str')
	mock_tf_load.assert_called_once()
	data = obj.load_label_map_from_json(path2label_map_json)
	assert type(data) == dict and len(data) == totalLabels

@patch('tensorflow.saved_model.load')
def test_load_image_into_numpy_array(mock_tf_load):
	obj = inference(path2label_map_json,'model_path:str')
	mock_tf_load.assert_called_once()

	data = obj.load_image_into_numpy_array(path2img)
	assert type(data) == np.ndarray
	assert data.shape == image_shape

@patch('tensorflow.saved_model.load')
def test_get_json_input(mock_tf_load):
	obj = inference(path2label_map_json,'model_path:str')
	mock_tf_load.assert_called_once()

	request = obj._get_json_input(path2img)
	assert type(request) == dict

@patch('tensorflow.saved_model.load')
def test_get_spec_pics_for_wav(mock_tf_load):
	obj = inference(path2label_map_json,'model_path:str')
	mock_tf_load.assert_called_once()

	os.mkdir(path)
	obj.get_spec_pics_for_wav(path2wav, path)
	os.remove(os.path.join(path+'/cv/'+'E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg'))
	os.remove(os.path.join(path+'/stft/'+'E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg'))
	shutil.rmtree(os.path.join(path), ignore_errors=True)

	with pytest.raises(ValueError, match=r'not a wav file'):
		obj.get_spec_pics_for_wav(path2img , path)

@patch('tensorflow.saved_model.load')
def test_nms(mock_tf_load):
	obj = inference(path2label_map_json,'model_path:str')
	assert obj.nms_th == 0.5
	mock_tf_load.assert_called_once()

	rects = [(np.asarray([x1,y1,x2,y2]), np.asarray(75), np.asarray(1)), (np.asarray([x1,y1,x2,y2*2]), np.asarray(85), np.asarray(2))]

	boxes, scores, classes = obj.nms(rects)
	assert type(boxes) == np.ndarray and type(scores) == np.ndarray and type(classes) == np.ndarray
	assert np.array_equal(boxes, np.asarray([[x1,y1,x2,y2*2]]))
	assert scores == 85 and classes == 2

@patch('tensorflow.saved_model.load')
def test_intersection(mock_tf_load):
	obj = inference(path2label_map_json,'model_path:str')
	mock_tf_load.assert_called_once()

	rects = [(np.asarray([x1,y1,x2,y2]), np.asarray(75), np.asarray(1)), (np.asarray([x1,y1,x2,y2*2]), np.asarray(85), np.asarray(2))]
	output = obj.intersection(rects[0][0], rects[1][0])
	assert type(output) == np.int32

@patch('tensorflow.saved_model.load')
def test_square(mock_tf_load):
	obj = inference(path2label_map_json,'model_path:str')
	mock_tf_load.assert_called_once()

	rects = [(np.asarray([x1,y1,x2,y2]), np.asarray(75), np.asarray(1)), (np.asarray([x1,y1,x2,y2*2]), np.asarray(85), np.asarray(2))]
	output = obj.square(rects[0][0])
	assert type(output) == np.int32

def test_process_detections():
	pass

def test_inference_as_raw_output():
	pass

def test_inference_one_img_with_plot():
	pass

def test_inference_one_img_with_plot_on_stft():
	pass