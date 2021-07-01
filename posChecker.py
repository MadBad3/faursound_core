import pandas as pd
import numpy as np
from typing import List
import cv2 as cv


class posChecker(object):
    def __init__(self) -> None:
        pass


    def _get_cut_center_position(self,stft_pic_path:str, x1:float,y1:float,x2:float,y2:float):
        img = cv.imread(stft_pic_path)
        y_height, x_width = img.shape

        center_position = ((x2-x1)/x_width,(y2-y1)/y_height)
        return center_position


    def _check_detection_position(self, stft_pic_path:str, x1:float,y1:float,x2:float,y2:float, label:str):
        center_position_x, center_position_y = self._get_cut_center_position(stft_pic_path,
                                                                            x1=x1,y1=y1,x2=x2,y2=y2)
        if label == 'blocking':
            output = self._check_pos_blocking(center_position_x, center_position_y)

        elif label == 'pain':
            output = self._check_pos_pain(center_position_x, center_position_y)

        elif label == 'hooting':
            output = self._check_pos_hooting(center_position_x, center_position_y)    

        elif label == 'modulation':
            output = self._check_pos_modulation(center_position_x, center_position_y)

        elif label == 'grinding':
            output = self._check_pos_grinding(center_position_x, center_position_y)

        elif label == 'over_running':
            output = self._check_pos_over_running(center_position_x, center_position_y)

        elif label == 'tic':
            output = self._check_pos_tic(center_position_x, center_position_y)
            
        elif label == 'knock':
            output = self._check_pos_knock(center_position_x, center_position_y)

        elif label == 'scratching':
            output = self._check_pos_scratching(center_position_x, center_position_y)
            
        elif label == 'spike':
            output = self._check_pos_spike(center_position_x, center_position_y=center_position_x, center_position_y,score=score)
            
        elif label == 'bench':
            output = self._check_pos_bench(center_position_x, center_position_y)
            
        elif label == 'measurement_issue':
            output = self._check_pos_measurement_issue(center_position_x, center_position_y)
            
        elif label == 'rattle_product':
            output = self._check_pos_rattle_product(center_position_x, center_position_y)
            
        elif label == 'buzzing':
            output = self._check_pos_buzzing(center_position_x, center_position_y)

        elif label == 'vibration_machine':
            output = self._check_pos_vibration_machine(center_position_x, center_position_y)

        else :
            raise ValueError('label is not inside of pre-defined list for CL calculation')

        return output 


    def _check_pos_blocking(self, center_position_x, center_position_y):
        blocking_position_range = {
            'center_x_min': 0.0,
            'center_y_min': 0.01,
            'center_x_max': 1.0,
            'center_y_max': 0.99,
        }
        pass