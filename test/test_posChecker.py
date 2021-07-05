import pytest, os, pandas as pd
import sys
sys.path.append((os.path.dirname(os.path.dirname(__file__))))
from posChecker import posChecker

test_folder_location = os.path.dirname(__file__)
output_folder = test_folder_location
stft_pic_path = os.path.join(output_folder+'/test/E-Tilt_6202-PC1_2402749_20210201_052215_11ACF9023_750N - Down.jpg')

x1 = float(174)
y1 = float(510)
x2 = float(300)
y2 = float(622)


def test_get_cut_covered_position():
	pc = posChecker(stft_pic_path)
	output = pc._get_cut_covered_position(x1, y1, x2, y2)
	assert type(output) == tuple
	assert output == ((x2-x1)/pc.x_width,(y2-y1)/pc.y_height)


def test_check_condition_position():
	pc = posChecker(stft_pic_path)

	blocking_position_range = {
		'covered_x_min': 0.12,
		'covered_y_min': 0.1,
		'covered_x_max': 0.24,
		'covered_y_max': 0.2,
		'y_height_up_limit': pc.y_height*0.5,
		'y_height_bottom_limit': pc.y_height*0.85
	}

	output = pc._check_condition_position(blocking_position_range, (x2-x1)/pc.x_width, (y2-y1)/pc.y_height, y1, y2)
	assert output == True

	output = pc._check_condition_position(blocking_position_range, ((x2*0.75)-x1)/pc.x_width, (y2-y1)/pc.y_height, y1, y2)
	assert output == False


def test_check_detection_position():
	pc = posChecker(stft_pic_path)

	with pytest.raises(ValueError, match=r'label is not inside of pre-defined list for CL calculation'):
		pc._check_detection_position(x1, y1, x2, y2, 'hoot')

	output = pc._check_detection_position(x1, y1, x2, y2, 'blocking')
	assert type(output) == str and output == 'pos_OK'

	output = pc._check_detection_position(x1, y1, x2, y2, 'hooting')
	assert type(output) == str and output == 'pos_NOK'


def test_check_pos_blocking():
	pc = posChecker(stft_pic_path)
	covered_position = pc._get_cut_covered_position(x1, y1, x2, y2)
	output = pc._check_pos_blocking(covered_position[0], covered_position[1], x1, y1, x2, y2)
	assert type(output) == str and output == 'pos_OK'

	output = pc._check_pos_blocking(covered_position[0], covered_position[1], x1, pc.y_height*0.4, x2, y2)
	assert type(output) == str and output == 'pos_NOK'


def test_check_pos_pain():
	pc = posChecker(stft_pic_path)

	x1 = float(6)
	y1 = float(479)
	x2 = float(171)
	y2 = float(629)

	covered_position = pc._get_cut_covered_position(x1, y1, x2, y2)
	output = pc._check_pos_pain(covered_position[0], covered_position[1], x1, y1, x2, y2)
	assert type(output) == str and output == 'pos_OK'

	output = pc._check_pos_pain(covered_position[0], covered_position[1], x1, pc.y_height*0.2, x2, y2)
	assert type(output) == str and output == 'pos_NOK'


def test_check_pos_hooting():
	pc = posChecker(stft_pic_path)

	x1 = float(404)
	y1 = float(518)
	x2 = float(968)
	y2 = float(540)

	covered_position = pc._get_cut_covered_position(x1, y1, x2, y2)
	output = pc._check_pos_hooting(covered_position[0], covered_position[1], x1, y1, x2, y2)
	assert type(output) == str and output == 'pos_OK'

	output = pc._check_pos_hooting(covered_position[0], covered_position[1], x1, pc.y_height*0.4, x2, y2)
	assert type(output) == str and output == 'pos_NOK'


def test_check_pos_modulation():
	pc = posChecker(stft_pic_path)

	x1 = float(219)
	y1 = float(586)
	x2 = float(943)
	y2 = float(627)

	covered_position = pc._get_cut_covered_position(x1, y1, x2, y2)
	output = pc._check_pos_modulation(covered_position[0], covered_position[1], x1, y1, x2, y2)
	assert type(output) == str and output == 'pos_OK'

	output = pc._check_pos_modulation(covered_position[0], covered_position[1], x1, pc.y_height*0.4, x2, y2)
	assert type(output) == str and output == 'pos_NOK'


def test_check_pos_grinding():
	pc = posChecker(stft_pic_path)

	x1 = float(1)
	y1 = float(473)
	x2 = float(968)
	y2 = float(629)

	covered_position = pc._get_cut_covered_position(x1, y1, x2, y2)
	output = pc._check_pos_grinding(covered_position[0], covered_position[1], x1, y1, x2, y2)
	assert type(output) == str and output == 'pos_OK'

	output = pc._check_pos_grinding(covered_position[0], covered_position[1], x1, y1, x2, pc.y_height*0.86)
	assert type(output) == str and output == 'pos_NOK'


def test_check_pos_over_running():
	pc = posChecker(stft_pic_path)

	x1 = float(518)
	y1 = float(148)
	x2 = float(604)
	y2 = float(379)

	covered_position = pc._get_cut_covered_position(x1, y1, x2, y2)
	output = pc._check_pos_over_running(covered_position[0], covered_position[1], x1, y1, x2, y2)
	assert type(output) == str and output == 'pos_OK'

	output = pc._check_pos_over_running(covered_position[0], covered_position[1], x1, y1, x2, pc.y_height*0.66)
	assert type(output) == str and output == 'pos_NOK'


def test_check_pos_tic():
	pc = posChecker(stft_pic_path)

	x1 = float(1)
	y1 = float(325)
	x2 = float(968)
	y2 = float(460)

	covered_position = pc._get_cut_covered_position(x1, y1, x2, y2)
	output = pc._check_pos_tic(covered_position[0], covered_position[1], x1, y1, x2, y2)
	assert type(output) == str and output == 'pos_OK'

	output = pc._check_pos_tic(covered_position[0], covered_position[1], x1, y1, x2, pc.y_height*0.66)
	assert type(output) == str and output == 'pos_NOK'


def test_check_pos_knock():
	pc = posChecker(stft_pic_path)

	x1 = float(17)
	y1 = float(472)
	x2 = float(939)
	y2 = float(706)

	covered_position = pc._get_cut_covered_position(x1, y1, x2, y2)
	output = pc._check_pos_knock(covered_position[0], covered_position[1], x1, y1, x2, y2)
	assert type(output) == str and output == 'pos_OK'

	output = pc._check_pos_knock(covered_position[0], covered_position[1], x1, y1, x2, pc.y_height*0.95)
	assert type(output) == str and output == 'pos_NOK'


def test_check_pos_scratching():
	pc = posChecker(stft_pic_path)

	x1 = float(1)
	y1 = float(1)
	x2 = float(658)
	y2 = float(150)

	covered_position = pc._get_cut_covered_position(x1, y1, x2, y2)
	output = pc._check_pos_scratching(covered_position[0], covered_position[1], x1, y1, x2, y2)
	assert type(output) == str and output == 'pos_OK'

	output = pc._check_pos_scratching(covered_position[0], covered_position[1], x1, y1, x2, pc.y_height*0.25)
	assert type(output) == str and output == 'pos_NOK'

def test_check_pos_spike():
	pc = posChecker(stft_pic_path)

	x1 = float(97)
	y1 = float(1)
	x2 = float(117)
	y2 = float(684)

	covered_position = pc._get_cut_covered_position(x1, y1, x2, y2)
	output = pc._check_pos_spike(covered_position[0], covered_position[1], x1, y1, x2, y2)
	assert type(output) == str and output == 'pos_OK'

	output = pc._check_pos_spike(covered_position[0], covered_position[1]/4, x1, y1, x2, y2)
	assert type(output) == str and output == 'pos_NOK'


def test_check_pos_bench():
	pc = posChecker(stft_pic_path)

	x1 = float(427)
	y1 = float(727)
	x2 = float(466)
	y2 = float(747)

	covered_position = pc._get_cut_covered_position(x1, y1, x2, y2)
	output = pc._check_pos_bench(covered_position[0], covered_position[1], x1, y1, x2, y2)
	assert type(output) == str and output == 'pos_OK'

	output = pc._check_pos_bench(covered_position[0], covered_position[1], x1, pc.y_height*0.8, x2, y2)
	assert type(output) == str and output == 'pos_NOK'


def test_check_pos_measurement_issue():
	pc = posChecker(stft_pic_path)

	x1 = float(250)
	y1 = float(1)
	x2 = float(508)
	y2 = float(744)

	covered_position = pc._get_cut_covered_position(x1, y1, x2, y2)
	output = pc._check_pos_measurement_issue(covered_position[0], covered_position[1], x1, y1, x2, y2)
	assert type(output) == str and output == 'pos_OK'

	output = pc._check_pos_measurement_issue(covered_position[0], covered_position[1]/4, x1, y1, x2, y2)
	assert type(output) == str and output == 'pos_NOK'


def test_check_pos_rattle_product():
	pc = posChecker(stft_pic_path)

	x1 = float(175)
	y1 = float(726)
	x2 = float(968)
	y2 = float(757)

	covered_position = pc._get_cut_covered_position(x1, y1, x2, y2)
	output = pc._check_pos_rattle_product(covered_position[0], covered_position[1], x1, y1, x2, y2)
	assert type(output) == str and output == 'pos_OK'

	output = pc._check_pos_rattle_product(covered_position[0], covered_position[1], x1, pc.y_height*0.8, x2, y2)
	assert type(output) == str and output == 'pos_NOK'


def test_check_pos_buzzing():
	pc = posChecker(stft_pic_path)

	x1 = float(1)
	y1 = float(205)
	x2 = float(730)
	y2 = float(231)

	covered_position = pc._get_cut_covered_position(x1, y1, x2, y2)
	output = pc._check_pos_buzzing(covered_position[0], covered_position[1], x1, y1, x2, y2)
	assert type(output) == str and output == 'pos_OK'

	output = pc._check_pos_buzzing(covered_position[0], covered_position[1], x1, pc.y_height*0.15, x2, y2)
	assert type(output) == str and output == 'pos_NOK'


def test_check_pos_vibration_machine():
	pass