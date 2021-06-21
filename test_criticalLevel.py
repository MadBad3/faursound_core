import pytest, os, pandas as pd
import sys
# sys.path.append((os.path.dirname(os.path.dirname(__file__))))
from criticalLevel import criticalLevel

output_folder = r'./test'
label = 'blocking'
x1 = float(1)
y1 = float(250)
x2 = float(50)
y2 = float(300)
score = float(0.8)
window_size = 5
step = 1

def test_run():
	cl = criticalLevel()
	stft_pic_path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	output = cl.run(stft_pic_path, x1,y1,x2,y2,score, label)
	assert type(output) == int

def test_read_pic_and_box_to_array():
	stft_pic_path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	cl = criticalLevel()
	cutted_img = cl._read_pic_and_box_to_array(stft_pic_path, x1,y1,x2,y2)
	assert cutted_img.shape[1] == int(x2-x1) and cutted_img.shape[0] == int(y2-y1)

def test_calculate_critical_level():
	stft_pic_path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	cl = criticalLevel()
	cutted_img = cl._read_pic_and_box_to_array(stft_pic_path, x1,y1,x2,y2)
	output = cl.calculate_critical_level(label, score, cutted_img)
	assert type(output) == int

	with pytest.raises(ValueError, match=r'label is not inside of pre-defined list for CL calculation'):
		cl.calculate_critical_level('label',score, cutted_img)

def test_calcul_cl_blocking():
	stft_pic_path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	cl = criticalLevel()
	cutted_img = cl._read_pic_and_box_to_array(stft_pic_path, x1,y1,x2,y2)
	output = cl._calcul_cl_blocking(cutted_img)
	assert type(output) == int

def test_calcul_cl_pain():
	stft_pic_path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	cl = criticalLevel()
	cutted_img = cl._read_pic_and_box_to_array(stft_pic_path, x1,y1,x2,y2)
	output = cl._calcul_cl_pain(cutted_img)
	assert type(output) == int

def test_calcul_cl_hooting():
	stft_pic_path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	cl = criticalLevel()
	cutted_img = cl._read_pic_and_box_to_array(stft_pic_path, x1,y1,x2,y2)
	output = cl._calcul_cl_hooting(cutted_img)
	assert type(output) == int

def test_mean_in_y_then_mean_in_x():
	stft_pic_path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	cl = criticalLevel()
	cutted_img = cl._read_pic_and_box_to_array(stft_pic_path, x1,y1,x2,y2)
	output = cl._mean_in_y_then_mean_in_x(cutted_img)
	assert type(output) == int

def test_calcul_cl_modulation():
	stft_pic_path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	cl = criticalLevel()
	cutted_img = cl._read_pic_and_box_to_array(stft_pic_path, x1,y1,x2,y2)
	output = cl._calcul_cl_modulation(cutted_img)
	assert type(output) == int

def test_calculate_moving_average():
	stft_pic_path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	cl = criticalLevel()
	cutted_img = cl._read_pic_and_box_to_array(stft_pic_path, x1,y1,x2,y2)
	example = {'Type': ['example1','example2'], 
			'Price': [22000,25000]}

	df = pd.DataFrame(example)
	output = cl._calculate_moving_average(df, window_size=window_size, step=step)
	assert type(output) == pd.core.frame.DataFrame

def test_max_in_y_then_mean_in_x():
	stft_pic_path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	cl = criticalLevel()
	cutted_img = cl._read_pic_and_box_to_array(stft_pic_path, x1,y1,x2,y2)
	output = cl._max_in_y_then_mean_in_x(cutted_img)
	assert type(output) == int

def test_calcul_cl_grinding():
	stft_pic_path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	cl = criticalLevel()
	cutted_img = cl._read_pic_and_box_to_array(stft_pic_path, x1,y1,x2,y2)
	output = cl._calcul_cl_grinding(cutted_img)
	assert type(output) == int

def test_calcul_cl_over_running():
	stft_pic_path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	cl = criticalLevel()
	cutted_img = cl._read_pic_and_box_to_array(stft_pic_path, x1,y1,x2,y2)
	output = cl._calcul_cl_over_running(cutted_img)
	assert type(output) == int

def test_calcul_cl_tic():
	stft_pic_path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	cl = criticalLevel()
	cutted_img = cl._read_pic_and_box_to_array(stft_pic_path, x1,y1,x2,y2)
	output = cl._calcul_cl_tic(cutted_img)
	assert type(output) == int

def test_calcul_cl_knock():
	stft_pic_path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	cl = criticalLevel()
	cutted_img = cl._read_pic_and_box_to_array(stft_pic_path, x1,y1,x2,y2)
	output = cl._calcul_cl_knock(cutted_img)
	assert type(output) == int

def test_calcul_cl_scratching():
	stft_pic_path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	cl = criticalLevel()
	cutted_img = cl._read_pic_and_box_to_array(stft_pic_path, x1,y1,x2,y2)
	output = cl._calcul_cl_scratching(cutted_img)
	assert type(output) == int

def test_calcul_cl_spike():
	stft_pic_path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	cl = criticalLevel()
	cutted_img = cl._read_pic_and_box_to_array(stft_pic_path, x1,y1,x2,y2)
	output = cl._calcul_cl_spike(cutted_img,score)
	assert type(output) == int

def test_calcul_cl_bench():
	stft_pic_path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	cl = criticalLevel()
	cutted_img = cl._read_pic_and_box_to_array(stft_pic_path, x1,y1,x2,y2)
	output = cl._calcul_cl_bench(cutted_img)
	assert type(output) == int

def test_calcul_cl_knock():
	stft_pic_path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	cl = criticalLevel()
	cutted_img = cl._read_pic_and_box_to_array(stft_pic_path, x1,y1,x2,y2)
	output = cl._calcul_cl_knock(cutted_img)
	assert type(output) == int

def test_calcul_cl_measurement_issue():
	stft_pic_path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	cl = criticalLevel()
	cutted_img = cl._read_pic_and_box_to_array(stft_pic_path, x1,y1,x2,y2)
	output = cl._calcul_cl_measurement_issue(cutted_img)
	assert type(output) == int

def test_calcul_cl_rattle_product():
	stft_pic_path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	cl = criticalLevel()
	cutted_img = cl._read_pic_and_box_to_array(stft_pic_path, x1,y1,x2,y2)
	output = cl._calcul_cl_rattle_product(cutted_img)
	assert type(output) == int

def test_calcul_cl_buzzing():
	stft_pic_path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	cl = criticalLevel()
	cutted_img = cl._read_pic_and_box_to_array(stft_pic_path, x1,y1,x2,y2)
	output = cl._calcul_cl_buzzing(cutted_img)
	assert type(output) == int

def test_calcul_cl_vibration_machine():
	stft_pic_path = os.path.join(output_folder+'/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')
	cl = criticalLevel()
	cutted_img = cl._read_pic_and_box_to_array(stft_pic_path, x1,y1,x2,y2)
	output = cl._calcul_cl_vibration_machine(cutted_img)
	assert type(output) == int