import pandas as pd
import numpy as np
from typing import List
import cv2 as cv
from itertools import count
import os
import sys
sys.path.append((os.path.dirname(os.path.dirname(__file__))))

from faursound_core.etiltWav import etiltWav
from itertools import count


class criticalLevel(object):

    def __init__(self):
        self.num_spike = count(1)

    def run(self,stft_pic_path:str, x1:float,y1:float,x2:float,y2:float, label:str) -> int:
        """run critical level calculation

        :param stft_pic_path: [stft picture path]
        :type stft_pic_path: str
        :param x1: [x1]
        :type x1: float
        :param y1: [y1]
        :type y1: float
        :param x2: [x2]
        :type x2: float
        :param y2: [y2]
        :type y2: float
        :param label: [label]
        :type label: str
        :return: [critical level value]
        :rtype: int
        """
        # just in case x1,y1,x2,y2 is not float.
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)

        cutted_img = self._read_pic_and_box_to_array(stft_pic_path, x1,y1,x2,y2)
        output = self.calculate_critical_level(label = label, cutted_img = cutted_img)
        return output


    def _read_pic_and_box_to_array(self, stft_pic_path:str, x1:float,y1:float,x2:float,y2:float) -> np.ndarray:

        img = cv.imread(stft_pic_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cutted_img = gray[int(y1):int(y2),int(x1):int(x2)]
        return cutted_img


    def _calculate_moving_average(self, df:pd.DataFrame, window_size:int, step:int) -> pd.DataFrame:
        start = window_size - 1 ## it is the index of your 1st valid value.
        df = df.rolling(window_size).mean()[start::step]
        return df


    def calculate_critical_level(self, label:str, cutted_img:np.ndarray):
        if label == 'blocking':
            output = self._calcul_cl_blocking(cutted_img)

        elif label == 'pain':
            output = self._calcul_cl_pain(cutted_img)

        elif label == 'hooting':
            output = self._calcul_cl_hooting(cutted_img)    

        elif label == 'modulation':
            output = self._calcul_cl_modulation(cutted_img)

        elif label == 'grinding':
            output = self._calcul_cl_grinding(cutted_img)

        elif label == 'over_running':
            output = self._calcul_cl_over_running(cutted_img)

        elif label == 'tic':
            output = self._calcul_cl_tic(cutted_img)
            
        elif label == 'knock':
            output = self._calcul_cl_knock(cutted_img)

        elif label == 'scratching':
            output = self._calcul_cl_scratching(cutted_img)
            
        elif label == 'spike':
            output = self._calcul_cl_spike(cutted_img)
            
        elif label == 'bench':
            output = self._calcul_cl_bench(cutted_img)
            
        elif label == 'measurement_issue':
            output = self._calcul_cl_measurement_issue(cutted_img)
            
        elif label == 'rattle_product':
            output = self._calcul_cl_rattle_product(cutted_img)
            
        elif label == 'buzzing':
            output = self._calcul_cl_buzzing(cutted_img)

        elif label == 'vibration_machine':
            output = self._calcul_cl_vibration_machine(cutted_img)

        else :
            raise ValueError('label is not inside of pre-defined list for CL calculation')

        return output

            
    def _calcul_cl_blocking(self,cutted_img:np.ndarray) -> int:
        """critical lever calculation for blocking
        #? increase range * ampitude

        :param cutted_img: [cutted image array from bounding box]
        :type cutted_img: np.ndarray
        :return: [calculated cl value]
        :rtype: [int]
        """
        y_max_pos = np.argmax(cutted_img,axis = 0)
        increase_range = y_max_pos.max() - y_max_pos.min()
        amp = self._max_in_y_then_mean_in_x(cutted_img)
        return int(increase_range * amp / 100)              

    def _calcul_cl_pain(self,cutted_img:np.ndarray) -> int:
        """critical lever calculation for pain
        #? decrease range * ampitude

        :param cutted_img: [cutted image array from bounding box]
        :type cutted_img: np.ndarray
        :return: [calculated cl value]
        :rtype: [int]
        """
        y_max_pos = np.argmax(cutted_img,axis = 0)
        decrease_range = y_max_pos.max() - y_max_pos.min()
        amp = self._max_in_y_then_mean_in_x(cutted_img)
        return int(decrease_range * amp / 100)             

    def _calcul_cl_hooting(self,cutted_img:np.ndarray) -> int:
        """critical lever calculation for hooting
        #? mean_in_y_then_mean_in_x

        :param cutted_img: [cutted image array from bounding box]
        :type cutted_img: np.ndarray
        :return: [calculated cl value]
        :rtype: [int]
        """
        return self._mean_in_y_then_mean_in_x(cutted_img)

    def _calcul_cl_modulation(self,cutted_img:np.ndarray) -> int:
        """critical lever calculation for modulation
        #? mod_heigh * ampitude
        #? to be update ! y_max_pos should be smoothed

        :param cutted_img: [cutted image array from bounding box]
        :type cutted_img: np.ndarray
        :return: [calculated cl value]
        :rtype: [int]
        """
        y_max_pos = np.argmax(cutted_img,axis = 0)
        df = pd.DataFrame({'y_max_pos': y_max_pos})
        df = self._calculate_moving_average(df, window_size=5, step=1)
        
        mod_heigh = df['y_max_pos'].max() - df['y_max_pos'].min()

        amp = self._max_in_y_then_mean_in_x(cutted_img)
        return int(mod_heigh * amp / 100)                    

    def _calcul_cl_grinding(self,cutted_img:np.ndarray) -> int:
        """critical lever calculation for grinding
        #? to be checked

        :param cutted_img: [cutted image array from bounding box]
        :type cutted_img: np.ndarray
        :return: [calculated cl value]
        :rtype: [int]
        """
        mean_in_x = np.mean(cutted_img,axis = 1) # mean value for each row

        rms = np.sqrt(np.mean(mean_in_x**2)) # calcualte RMS
        return int(rms)                     

    def _calcul_cl_over_running(self,cutted_img:np.ndarray) -> int:
        """critical lever calculation for over running
        #? width ?

        :param cutted_img: [cutted image array from bounding box]
        :type cutted_img: np.ndarray
        :return: [calculated cl value]
        :rtype: [int]
        """
        width = cutted_img.shape[1]
        return width                                  

    def _calcul_cl_tic(self,cutted_img:np.ndarray) -> int:
        """critical lever calculation for tic
        #? max_in_y_then_mean_in_x ?

        :param cutted_img: [cutted image array from bounding box]
        :type cutted_img: np.ndarray
        :return: [calculated cl value]
        :rtype: [int]
        """
        return self._max_in_y_then_mean_in_x(cutted_img) 

    def _calcul_cl_knock(self,cutted_img:np.ndarray) -> int:
        """critical lever calculation for knock
        #? max_in_y_then_mean_in_x ?

        :param cutted_img: [cutted image array from bounding box]
        :type cutted_img: np.ndarray
        :return: [calculated cl value]
        :rtype: [int]
        """
        return self._max_in_y_then_mean_in_x(cutted_img)

    def _calcul_cl_scratching(self,cutted_img:np.ndarray) -> int:
        """critical lever calculation for scratching
        #? high_amp_values's mean

        :param cutted_img: [cutted image array from bounding box]
        :type cutted_img: np.ndarray
        :return: [calculated cl value]
        :rtype: [int]
        """
        max_in_y = np.mean(cutted_img,axis = 0) # mean value for each column
        high_amp_values = max_in_y[max_in_y > max_in_y.mean()]
        return int(high_amp_values.mean())             

    def _calcul_cl_spike(self,cutted_img:np.ndarray) -> int:
        """critical lever calculation for spike
        #? spike's count

        :param cutted_img: [cutted image array from bounding box]
        :type cutted_img: np.ndarray
        :return: [calculated cl value]
        :rtype: [int]
        """
        # return int(cutted_img.shape[1] * cutted_img.shape[0] / 100) # just spike's area
        return next(self.num_spike)

    def _calcul_cl_bench(self,cutted_img:np.ndarray) -> int:
        """critical lever calculation for bench
        #? 0 as not from product

        :param cutted_img: [cutted image array from bounding box]
        :type cutted_img: np.ndarray
        :return: [calculated cl value]
        :rtype: [int]
        """
        return 1
    
    def _calcul_cl_measurement_issue(self,cutted_img:np.ndarray) -> int:
        """critical lever calculation for measurement_issue
        #? 0 as not from product

        :param cutted_img: [cutted image array from bounding box]
        :type cutted_img: np.ndarray
        :return: [calculated cl value]
        :rtype: [int]
        """
        return 1

    def _calcul_cl_rattle_product(self,cutted_img:np.ndarray) -> int:
        """critical lever calculation for rattle_product
        #? max_in_y_then_mean_in_x

        :param cutted_img: [cutted image array from bounding box]
        :type cutted_img: np.ndarray
        :return: [calculated cl value]
        :rtype: [int]
        """
        return self._max_in_y_then_mean_in_x(cutted_img)

    def _calcul_cl_buzzing(self,cutted_img:np.ndarray) -> int:
        """critical lever calculation for buzzing
        #? max_in_y_then_mean_in_x

        :param cutted_img: [cutted image array from bounding box]
        :type cutted_img: np.ndarray
        :return: [calculated cl value]
        :rtype: [int]
        """
        return self._max_in_y_then_mean_in_x(cutted_img)

    def _calcul_cl_vibration_machine(self,cutted_img:np.ndarray) -> int:
        """critical lever calculation for vibration_machine
        #? 0 as not from product

        :param cutted_img: [cutted image array from bounding box]
        :type cutted_img: np.ndarray
        :return: [calculated cl value]
        :rtype: [int]
        """
        return 1


    def _max_in_y_then_mean_in_x(self,cutted_img:np.ndarray) -> int:
        """calculation max in y axis then calculate mean in x axis

        :param cutted_img: [cutted image array from bounding box]
        :type cutted_img: np.ndarray
        :return: [calculated value]
        :rtype: int
        """
        max_in_y = np.max(cutted_img,axis = 0) # max value for each column
        return int(np.mean(max_in_y))               # return mean

    def _mean_in_y_then_mean_in_x(self,cutted_img:np.ndarray) -> int:
        """calculation mean in y axis then calculate mean in x axis

        :param cutted_img: [cutted image array from bounding box]
        :type cutted_img: np.ndarray
        :return: [calculated value]
        :rtype: int
        """
        mean_in_y = np.mean(cutted_img,axis = 0) # mean value for each column
        return int(np.mean(mean_in_y))               # return mean