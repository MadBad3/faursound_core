import pandas as pd
import numpy as np
from typing import List
import cv2 as cv


class posChecker(object):
    def __init__(self, stft_pic_path:str) -> None:
        img = cv.imread(stft_pic_path)
        self.y_height, self.x_width = img.shape


    def _get_cut_covered_position(self, x1:float, y1:float, x2:float, y2:float) -> None:
        covered_position = ((x2-x1)/self.x_width,(y2-y1)/self.y_height)
        #self.covered_position_x = covered_position[0]
        #self.covered_position_y = covered_position[1]
        return covered_position


    def _check_condition_position(self, position_range:dict, covered_position_x:float, covered_position_y:float) -> bool:
        if((covered_position_x >= position_range.get('covered_x_min') and covered_position_x <= position_range.get('covered_x_max')) and 
            (covered_position_y >= position_range.get('covered_y_min') and covered_position_y <= position_range.get('covered_y_max')) and 
            (y1 > position_range.get('y_height_up_limit') and y2 < position_range.get('y_height_bottom_limit'))) :
            return True
        else:
            return False


    def _check_detection_position(self, x1:float, y1:float, x2:float, y2:float, label:str):
        covered_position_x, covered_position_y = self._get_cut_covered_position(x1=x1,y1=y1,x2=x2,y2=y2)

        if label == 'blocking':
            output = self._check_pos_blocking(covered_position_x, covered_position_y, x1, y1, x2, y2)

        elif label == 'pain':
            output = self._check_pos_pain(covered_position_x, covered_position_y, x1, y1, x2, y2)

        elif label == 'hooting':
            output = self._check_pos_hooting(covered_position_x, covered_position_y, x1, y1, x2, y2)    

        elif label == 'modulation':
            output = self._check_pos_modulation(covered_position_x, covered_position_y, x1, y1, x2, y2)

        elif label == 'grinding':
            output = self._check_pos_grinding(covered_position_x, covered_position_y, x1, y1, x2, y2)

        elif label == 'over_running':
            output = self._check_pos_over_running(covered_position_x, covered_position_y, x1, y1, x2, y2)

        elif label == 'tic':
            output = self._check_pos_tic(covered_position_x, covered_position_y, x1, y1, x2, y2)
            
        elif label == 'knock':
            output = self._check_pos_knock(covered_position_x, covered_position_y, x1, y1, x2, y2)

        elif label == 'scratching':
            output = self._check_pos_scratching(covered_position_x, covered_position_y, x1, y1, x2, y2)
            
        elif label == 'spike':
            output = self._check_pos_spike(covered_position_x, covered_position_y, score=score, x1, y1, x2, y2)
            
        elif label == 'bench':
            output = self._check_pos_bench(covered_position_x, covered_position_y, x1, y1, x2, y2)
            
        elif label == 'measurement_issue':
            output = self._check_pos_measurement_issue(covered_position_x, covered_position_y, x1, y1, x2, y2)
            
        elif label == 'rattle_product':
            output = self._check_pos_rattle_product(covered_position_x, covered_position_y, x1, y1, x2, y2)
            
        elif label == 'buzzing':
            output = self._check_pos_buzzing(covered_position_x, covered_position_y, x1, y1, x2, y2)

        elif label == 'vibration_machine':
            output = self._check_pos_vibration_machine(covered_position_x, covered_position_y, x1, y1, x2, y2)

        else :
            raise ValueError('label is not inside of pre-defined list for CL calculation')

        return output 


    def _check_pos_blocking(self, covered_position_x:float, covered_position_y:float, x1:float,y1:float,x2:float,y2:float) -> str:
        blocking_position_range = {
            'covered_x_min': 0.12,
            'covered_y_min': 0.1,
            'covered_x_max': 0.24,
            'covered_y_max': 0.2,
            'y_height_up_limit': self.y_height*0.5,
            'y_height_bottom_limit': self.y_height*0.85
        }
        
        if(self._check_condition_position(blocking_position_range, covered_position_x, covered_position_y)) :
            return 'pos_OK'
        else:
            return 'pos_NOK'


    def _check_pos_pain(self, covered_position_x:float, covered_position_y:float, x1:float,y1:float,x2:float,y2:float) -> str:
        pain_position_range = {
            'covered_x_min': 0.08,
            'covered_y_min': 0.05,
            'covered_x_max': 1.0,
            'covered_y_max': 0.26,
            'y_height_up_limit': self.y_height*0.3,
            'y_height_bottom_limit': self.y_height*0.85
        }

        if(self._check_condition_position(pain_position_range, covered_position_x, covered_position_y)) :
            return 'pos_OK'
        else:
            return 'pos_NOK'


    def _check_pos_hooting(self, covered_position_x:float, covered_position_y:float, x1:float,y1:float,x2:float,y2:float) -> str:
        hooting_position_range = {
            'covered_x_min': 0.67,
            'covered_y_min': 0.012,
            'covered_x_max': 1.0,
            'covered_y_max': 0.04,
            'y_height_up_limit': self.y_height*0.5,
            'y_height_bottom_limit': self.y_height*0.85
        }

        if(self._check_condition_position(hooting_position_range, covered_position_x, covered_position_y)) :
            return 'pos_OK'
        else:
            return 'pos_NOK'


    def _check_pos_modulation(self, covered_position_x:float, covered_position_y:float, x1:float,y1:float,x2:float,y2:float) -> str:
        modulation_position_range = {
            'covered_x_min': 0.105,
            'covered_y_min': 0.04,
            'covered_x_max': 1.0,
            'covered_y_max': 0.13,
            'y_height_up_limit': self.y_height*0.5,
            'y_height_bottom_limit': self.y_height*0.85
        }

        if(self._check_condition_position(modulation_position_range, covered_position_x, covered_position_y)) :
            return 'pos_OK'
        else:
            return 'pos_NOK'


    def _check_pos_grinding(self, covered_position_x:float, covered_position_y:float, x1:float,y1:float,x2:float,y2:float) -> str:
        grinding_position_range = {
            'covered_x_min': 0.95,
            'covered_y_min': 0.065,
            'covered_x_max': 1.0,
            'covered_y_max': 0.22,
            'y_height_up_limit': self.y_height*0.25,
            'y_height_bottom_limit': self.y_height*0.85
        }

        if(self._check_condition_position(grinding_position_range, covered_position_x, covered_position_y)) :
            return 'pos_OK'
        else:
            return 'pos_NOK'


    def _check_pos_over_running(self, covered_position_x:float, covered_position_y:float, x1:float,y1:float,x2:float,y2:float) -> str:
        over_running_position_range = {
            'covered_x_min': 0.2,
            'covered_y_min': 0.065,
            'covered_x_max': 1.0,
            'covered_y_max': 0.22,
            'y_height_up_limit': self.y_height*0.19,
            'y_height_bottom_limit': self.y_height*0.65
        }

        if(self._check_condition_position(over_running_position_range, covered_position_x, covered_position_y)) :
            return 'pos_OK'
        else:
            return 'pos_NOK'


    def _check_pos_tic(self, covered_position_x:float, covered_position_y:float, x1:float,y1:float,x2:float,y2:float) -> str:
        tic_position_range = {
            'covered_x_min': 0.5,
            'covered_y_min': 0.05,
            'covered_x_max': 1.0,
            'covered_y_max': 0.26,
            'y_height_up_limit': self.y_height*0.19,
            'y_height_bottom_limit': self.y_height*0.65
        }

        if(self._check_condition_position(tic_position_range, covered_position_x, covered_position_y)) :
            return 'pos_OK'
        else:
            return 'pos_NOK'


    def _check_pos_knock(self, covered_position_x:float, covered_position_y:float, x1:float,y1:float,x2:float,y2:float) -> str:
        knock_position_range = {
            'covered_x_min': 0.5,
            'covered_y_min': 0.2,
            'covered_x_max': 1.0,
            'covered_y_max': 0.4,
            'y_height_up_limit': self.y_height*0.5,
            'y_height_bottom_limit': self.y_height*0.94
        }

        if(self._check_condition_position(knock_position_range, covered_position_x, covered_position_y)) :
            return 'pos_OK'
        else:
            return 'pos_NOK'


    def _check_pos_scratching(self, covered_position_x:float, covered_position_y:float, x1:float,y1:float,x2:float,y2:float) -> str:
        scratching_position_range = {
            'covered_x_min': 0.25,
            'covered_y_min': 0.1,
            'covered_x_max': 1.0,
            'covered_y_max': 0.2,
            'y_height_up_limit': self.y_height*0.0,
            'y_height_bottom_limit': self.y_height*0.2
        }

        if(self._check_condition_position(scratching_position_range, covered_position_x, covered_position_y)) :
            return 'pos_OK'
        else:
            return 'pos_NOK'


    def _check_pos_spike(self, covered_position_x:float, covered_position_y:float, x1:float,y1:float,x2:float,y2:float) -> str:
        spike_position_range = {
            'covered_x_min': 0.0125,
            'covered_y_min': 0.3,
            'covered_x_max': 0.055,
            'covered_y_max': 1.0,
            'y_height_up_limit': self.y_height*0.0,
            'y_height_bottom_limit': self.y_height*1.0
        }

        if(self._check_condition_position(spike_position_range, covered_position_x, covered_position_y)) :
            return 'pos_OK'
        else:
            return 'pos_NOK'

    
    def _check_pos_bench(self, covered_position_x:float, covered_position_y:float, x1:float,y1:float,x2:float,y2:float) -> str:
        bench_position_range = {
            'covered_x_min': 0.045,
            'covered_y_min': 0.03,
            'covered_x_max': 0.13,
            'covered_y_max': 0.07,
            'y_height_up_limit': self.y_height*0.9,
            'y_height_bottom_limit': self.y_height*1.0
        }

        if(self._check_condition_position(bench_position_range, covered_position_x, covered_position_y)) :
            return 'pos_OK'
        else:
            return 'pos_NOK'


    def _check_pos_measurement_issue(self, covered_position_x:float, covered_position_y:float, x1:float,y1:float,x2:float,y2:float) -> str:
        measurement_issue_position_range = {
            'covered_x_min': 0.1,
            'covered_y_min': 0.5,
            'covered_x_max': 1.0,
            'covered_y_max': 1.0,
            'y_height_up_limit': self.y_height*0.0,
            'y_height_bottom_limit': self.y_height*1.0
        }

        if(self._check_condition_position(measurement_issue_position_range, covered_position_x, covered_position_y)) :
            return 'pos_OK'
        else:
            return 'pos_NOK'


    def _check_pos_rattle_product(self, covered_position_x:float, covered_position_y:float, x1:float,y1:float,x2:float,y2:float) -> str:
        rattle_product_position_range = {
            'covered_x_min': 0.66,
            'covered_y_min': 0.03,
            'covered_x_max': 1.0,
            'covered_y_max': 0.07,
            'y_height_up_limit': self.y_height*0.9,
            'y_height_bottom_limit': self.y_height*1.0
        }

        if(self._check_condition_position(rattle_product_position_range, covered_position_x, covered_position_y)) :
            return 'pos_OK'
        else:
            return 'pos_NOK'


    def _check_pos_buzzing(self, covered_position_x:float, covered_position_y:float, x1:float,y1:float,x2:float,y2:float) -> str:
        buzzing_product_position_range = {
            'covered_x_min': 0.25,
            'covered_y_min': 0.03,
            'covered_x_max': 0.8,
            'covered_y_max': 0.1,
            'y_height_up_limit': self.y_height*0.2,
            'y_height_bottom_limit': self.y_height*0.4
        }

        if(self._check_condition_position(buzzing_product_position_range, covered_position_x, covered_position_y)) :
            return 'pos_OK'
        else:
            return 'pos_NOK'


    def _check_pos_vibration_machine(self, covered_position_x:float, covered_position_y:float, x1:float,y1:float,x2:float,y2:float) -> str:
        vibration_machine_product_position_range = {
            'covered_x_min': 0.0,
            'covered_y_min': 0.0,
            'covered_x_max': 0.0,
            'covered_y_max': 0.0,
            'y_height_up_limit': self.y_height*0.0,
            'y_height_bottom_limit': self.y_height*0.0
        }

        if(self._check_condition_position(vibration_machine_position_range, covered_position_x, covered_position_y)) :
            return 'pos_OK'
        else:
            return 'pos_NOK'