import os
import sys
# sys.path.append((os.path.dirname(os.path.dirname(__file__))))

from _wavNVH import _wavNVH
import re
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


class etiltWav(_wavNVH):
    """class to deal with etilt wav file, including preprocess needed for spec pics.
    """

    LABELS = [
            'blocking','pain','hooting','modulation','grinding',
            'over_running','tic','knock','scratching','spike','bench',
            'measurement_issue','rattle_product','buzzing','vibration_machine',
            ]

    LABEL_LIST_PLANT = [
                        'Clicking', 'Choc', 'Scratching','Grinding', 'Peine',
                        'VarVitesse', 'Buzzing', 'Rattle', 'Dying', 
                        'Roughness', 'Sharpness'
                        ]

    LABEL_LIST_ERIC = [
                        'Spike vib', 'Single', 'Bench', 'Scratch', 
                        'mod', 'peine', 'periodic', 'Grind', 'over',
                        ]
                    # single means choc only one time

    RANGE_AUDITION_CC = 140   ## this is parameter we use in adobe audition cc when look at spectrum
    VMAX = 20  ## VMAX for stft spec
    VMIN = VMAX - (RANGE_AUDITION_CC / 2) ## VMIN calculated according to VMAX and RANGE_AUDITION_CC

    THRESHOLD = 100  ## for opencv, value are selected for e-tilt
    THRESHOLD_2 = 30 ## for opencv, value are selected for e-tilt
    BLUR_SIZE = 5 ## for opencv, value are selected for e-tilt
    UPPER_CUT = 150 ## for opencv, cut spec into 3 part to do cv operation
    LOWER_CUT = 115  ## for opencv, cut spec into 3 part to do cv operation

    def __init__(self, file_path:str, n_fft:int , hop_length:int, get_infor_from_fn:bool = True) -> None:
        """init for e-tilt specificly.

        :param file_path: [wav file path]
        :type file_path: str
        :param n_fft: [number of n_fft, usually we use 512]
        :type n_fft: int
        :param hop_length: [hop_length, should be 25%/50%/100% of n_fft]
        :type hop_length: int
        """

        super().__init__(file_path=file_path, n_fft=n_fft, hop_length=hop_length)
        
        self.direction = None  #! if no direction read from wav file = error
        self.load = None
        self.sn = None
        if get_infor_from_fn:
            self._get_info_from_filename()

        self.spec_pic_path = None


    @classmethod
    def load_df_meta(cls, df: pd.DataFrame):
        cls._df_meta = df


    @property
    def df_meta(self):
        return type(self)._df_meta


    @property
    def label(self):
        return self._get_lables_based_on_fn()


    def _get_lables_based_on_fn(self) -> str:
        """this function use fn to check with df, then return string including all labels
        """

        if self.df_meta.empty:
            raise ValueError('please load df_meta first ! -> using cls.load_df_meta')
        else :
            df_selected_based_on_fn = self.df_meta[self.df_meta.Wave.str.contains(rf'.*{self.sn}.*{self.load}.*{self.direction}',
                                                     regex=True, na=False)]
            # this is incase of the wav file name not same as the one in df.

            df_selected_based_on_fn = df_selected_based_on_fn.iloc[0].to_frame().T # only select first row , incase there is more than 1 match !
            df_label = df_selected_based_on_fn[self.LABEL_LIST_ERIC]

            return df_label.notna().dot(df_label.columns + '-').str.rstrip('-').item()  ##if a Series with length 0, then the .index.item() will throw a ValueError.


    def _get_info_from_filename(self) -> None:
        """get direction / load / serials number infomation from filename according to e-tilt regex
        """
        SN = re.search('\_[A-Z0-9]{1}[0-9]{1}[A-Z]{3}[0-9]{3,4}', self.fn)
        self.sn = SN.group().strip('_')
        force = re.search('(\_[0-9]{1,2}0N -)|( [0-9]{1,2}0N)', self.fn)
        force = force.group()
        force = force.strip('_')
        force = force.strip('-')
        force = force.strip()
        self.load = force
        direction = re.search('Up|Down', self.fn)
        self.direction = direction.group()


    def stft_custom_spec_to_pic(self, output_folder: str, vmin: int = VMIN, vmax: int = VMAX,
                                figsize=_wavNVH.FIGSIZE, dpi = _wavNVH.DPI, suffix:str = ''):
        """this function plot the custom stft log spectrum

        :param output_folder: [output_folder]
        :type output_folder: str
        :param vmin: [vmin]
        :type vmin: int
        :param vmax: [vmax]
        :type vmax: int
        :param figsize: [figsize], defaults to _wavNVH.FIGSIZE
        :type figsize: tuple, optional
        :param dpi: [dpi], defaults to 100
        :type dpi: int, optional
        :param suffix: [suffix for output file name], defaults to ''
        :type suffix: str, optional
        :return: [save path for the output picture -> used for opencv preprocess]
        :rtype: str
        """
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        x_coord = self._coord_time(self.stft_spec.shape[1], sr = self.sr, hop_length=self._hop_length)
        y_coord = self._coord_fft_hz(self.stft_spec.shape[0], sr = self.sr)

        fig, ax = plt.subplots(figsize=figsize)

        cf = ax.pcolormesh(x_coord, y_coord, self.stft_spec, cmap='magma', vmax=vmax, vmin=vmin)
        
        ax.set_yscale('symlog', linthresh=1000, linscale=1 / np.e, base=np.e)
        
        title , _ = os.path.splitext(self.fn)
        plt.axis('off')
        save_path = os.path.join(output_folder, title + suffix + self.PIC_FORMAT)      
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0,transparent=False, dpi = dpi)

        print(f'spec saved as picture in {output_folder}, with name {title + suffix + self.PIC_FORMAT}')
        plt.close('all')
        self.spec_pic_path = save_path


    def plot_stft_custom_spec(self, vmin:int =VMIN , vmax:int= VMAX, figsize=_wavNVH.FIGSIZE) -> None:
        """this function plot the stft_custom log spectrum

        :param vmin: [vmin]
        :type vmin: int
        :param vmax: [vmax]
        :type vmax: int
        :param figsize: [figsize], defaults to _wavNVH.FIGSIZE
        :type figsize: tuple, optional
        """

        x_coord = self._coord_time(self.stft_spec.shape[1], sr = self.sr, hop_length=self._hop_length)
        y_coord = self._coord_fft_hz(self.stft_spec.shape[0], sr = self.sr)

        fig, ax = plt.subplots(figsize=figsize)

        cf = ax.pcolormesh(x_coord, y_coord, self.stft_spec, cmap='magma', vmax=vmax, vmin=vmin)
        
        ax.set_yscale('symlog', linthresh=1000, linscale=1 / np.e, base=np.e)
        
        yticks = [0, 200, 400,600,800,1000,2000,3000,4000,5000,6000,7000,8000]
        ax.set_yticks(yticks)

        ax.yaxis.set_major_formatter(ScalarFormatter())
        fig.colorbar(cf, format='%+2.f', ax=ax)
        
        title , _ = os.path.splitext(self.fn)
        plt.title(title)
        plt.show()


    def save_stft_cv_pic(self, output_folder: str, figsize=_wavNVH.FIGSIZE,suffix:str = '') -> None:
        """save stft cv pic after cv_preprocess

        :param output_folder: [output folder]
        :type output_folder: str
        :param figsize: [figsize], defaults to _wavNVH.FIGSIZE
        :type figsize: [type], optional
        :param suffix: [suffix], defaults to ''
        :type suffix: str, optional
        """

        if not self.spec_pic_path:
            print('please save stft spec picture first !!')
        else :
            if not os.path.isdir(output_folder):
                os.mkdir(output_folder)

            temp = os.path.basename(self.spec_pic_path)
            fn, _ = os.path.splitext(temp)

            cv_pic = self.cv_prepocess(self.spec_pic_path)

            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(cv_pic)
            ax.axis('off')

            plt.savefig(os.path.join(output_folder, fn + suffix + etiltWav.PIC_FORMAT),
                        bbox_inches='tight', pad_inches=0, transparent=False)
            
            print(f'save cv picture for {fn + suffix + etiltWav.PIC_FORMAT} -> DONE ')
            plt.close('all')


    def _cv_prepocess_up(self,image_path,lower_cut:int = LOWER_CUT,upper_cut:int = UPPER_CUT, blur_size: int = BLUR_SIZE,
                    threshold:int = THRESHOLD, threshold_2:int = THRESHOLD_2) -> np.ndarray:

        img = cv.imread(image_path)

        lower_cut_position = img.shape[0] - lower_cut

        upper_img = img[:upper_cut ,:,:].copy()
        middle_img = img[upper_cut:lower_cut_position ,:,:].copy()
        lower_img = img[lower_cut_position: ,:,:].copy()

        def cv_op_on_upper_img(upper_img):

            gray = cv.cvtColor(upper_img, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(gray, threshold * 0.7 , 255, type=3)

            max_limit = int(threshold * 1.4)   #* == do cv.equalizeHist until only 'max_limit' (instead of 255)

            hist,bins = np.histogram(thresh.flatten(),256 ,[0,256])
            cdf = hist.cumsum()

            cdf_m = np.ma.masked_equal(cdf,0)
            cdf_m = (cdf_m - cdf_m.min())*max_limit/(cdf_m.max()-cdf_m.min())
            cdf = np.ma.filled(cdf_m,0).astype('uint8')
            img2 = cdf[thresh]

            return img2

        def cv_op_on_middle_img(middle_img):

            blur = cv.blur(middle_img, (blur_size, blur_size), cv.BORDER_DEFAULT)

            kernel = np.array([[0,-1,0], 
                                [-1,5,-1], 
                                [0,-1,0]])
            sharpen = cv.filter2D(blur, -1, kernel)

            gray = cv.cvtColor(sharpen, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(gray, threshold *0.8 , 255, type=3)
            dst = cv.equalizeHist(thresh)

            ret2, thresh2 = cv.threshold(dst, threshold_2 , 255, type = 3)
            return thresh2

        def cv_op_on_lower_img(lower_img):

            gray = cv.cvtColor(lower_img, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(gray, threshold * 1.3 , 255, type=3)

            max_limit = int(threshold * 2.0)   #* == do cv.equalizeHist until only 'max_limit' (instead of 255)

            hist,bins = np.histogram(thresh.flatten(),256 ,[0,256])
            cdf = hist.cumsum()

            cdf_m = np.ma.masked_equal(cdf,0)
            cdf_m = (cdf_m - cdf_m.min())*max_limit/(cdf_m.max()-cdf_m.min())
            cdf = np.ma.filled(cdf_m,0).astype('uint8')
            img2 = cdf[thresh]
            return img2
            
        def merge_3_parts_of_imgs(upper_img,middle_img,lower_img):
            return np.concatenate((upper_img, middle_img, lower_img), axis=0)

        upper_img = cv_op_on_upper_img(upper_img)
        middle_img = cv_op_on_middle_img(middle_img)
        lower_img = cv_op_on_lower_img(lower_img)

        img_final_main = merge_3_parts_of_imgs(upper_img, middle_img, lower_img)
        color_img = cv.cvtColor(img_final_main, cv.COLOR_GRAY2RGB)
        return color_img


    def _cv_prepocess_down(self,image_path,lower_cut:int = LOWER_CUT,upper_cut:int = UPPER_CUT, blur_size: int = BLUR_SIZE,
                    threshold:int = THRESHOLD, threshold_2:int = THRESHOLD_2) -> np.ndarray:

        img = cv.imread(image_path)

        lower_cut_position = img.shape[0] - lower_cut

        upper_img = img[:upper_cut ,:,:].copy()
        middle_img = img[upper_cut:lower_cut_position ,:,:].copy()
        lower_img = img[lower_cut_position: ,:,:].copy()

        def cv_op_on_upper_img(upper_img):

            gray = cv.cvtColor(upper_img, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(gray, threshold * 0.6 , 255, type=3)

            max_limit = int(threshold * 1.2)   #* == do cv.equalizeHist until only 'max_limit' (instead of 255)

            hist,bins = np.histogram(thresh.flatten(),256 ,[0,256])
            cdf = hist.cumsum()

            cdf_m = np.ma.masked_equal(cdf,0)
            cdf_m = (cdf_m - cdf_m.min())*max_limit/(cdf_m.max()-cdf_m.min())
            cdf = np.ma.filled(cdf_m,0).astype('uint8')
            img2 = cdf[thresh]

            return img2

        def cv_op_on_middle_img(middle_img):

            blur = cv.blur(middle_img, (blur_size, blur_size), cv.BORDER_DEFAULT)

            kernel = np.array([[0,-1,0], 
                                [-1,5,-1], 
                                [0,-1,0]])
            sharpen = cv.filter2D(blur, -1, kernel)

            gray = cv.cvtColor(sharpen, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(gray, threshold , 255, type=3)
            dst = cv.equalizeHist(thresh)

            ret2, thresh2 = cv.threshold(dst, threshold_2 , 255, type = 3)
            return thresh2, gray

        def cv_op_on_lower_img(lower_img):

            gray = cv.cvtColor(lower_img, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(gray, threshold * 1.3 , 255, type=3)

            max_limit = int(threshold * 2.0)   #* == do cv.equalizeHist until only 'max_limit' (instead of 255)

            hist,bins = np.histogram(thresh.flatten(),256 ,[0,256])
            cdf = hist.cumsum()

            cdf_m = np.ma.masked_equal(cdf,0)
            cdf_m = (cdf_m - cdf_m.min())*max_limit/(cdf_m.max()-cdf_m.min())
            cdf = np.ma.filled(cdf_m,0).astype('uint8')
            img2 = cdf[thresh]
            return img2
            
        def merge_3_parts_of_imgs(upper_img,middle_img,lower_img):
            return np.concatenate((upper_img, middle_img, lower_img), axis=0)

        upper_img = cv_op_on_upper_img(upper_img)
        middle_img, middle_img_gray = cv_op_on_middle_img(middle_img)
        lower_img = cv_op_on_lower_img(lower_img)

        over_run_channel_middle = self.over_run_preprocessing(middle_img_gray)

        img_final_main = merge_3_parts_of_imgs(upper_img, middle_img, lower_img)
        img_over_run_channel = merge_3_parts_of_imgs(np.zeros_like(upper_img), over_run_channel_middle, np.zeros_like(lower_img))

        #* over running channel only done for Down direction !
        color_img = cv.cvtColor(img_final_main, cv.COLOR_GRAY2RGB)
        color_img[:,:,2] = color_img[:,:,2] + img_over_run_channel / 255 * 130

        return color_img


    def cv_prepocess(self, image_path):
        if self.direction == 'Down':
            color_img = self._cv_prepocess_down(image_path)
        elif self.direction == 'Up':
            color_img = self._cv_prepocess_up(image_path)
        else :
            raise KeyError('etiltWav have direction not up and down -> can not do cv_preprocess')
        return color_img


    def over_run_preprocessing(self, gray:np.ndarray,percentage_row:float = 0.35,percentage_column:float =0.15,sensitivity:float = 0.02) -> np.ndarray:
        def _get_idxs_according_to_axis_sum(a, axis, percentage, greater_than:bool):
            a_ = a / 255  # get all number 255 to 1
            if greater_than:
                idxs = np.where(np.sum(a_,axis = axis) > percentage * a_.shape[axis]) # return idxs for column meet target
            else:
                idxs = np.where(np.sum(a_,axis = axis) < percentage * a_.shape[axis]) # return idxs for column meet target
            return idxs

        def over_run_row_process(gray,percentage_row:float):
            # if row have too much white -> means from product functional -> we set all of them to 0
            ret, thresh = cv.threshold(gray, 50, 255, cv.THRESH_BINARY_INV)
            idxs = _get_idxs_according_to_axis_sum(thresh, axis=1, percentage = percentage_row, greater_than = True)

            row_output = thresh.copy()
            if len(idxs)>0:
                row_output[idxs,:] = 0 # set idx meet target to value 0
            return row_output, thresh


        def over_run_column_process(row_output,thresh, percentage_column):
            # if column have too little white -> means not over running -> we set all of them to 0
            idxs = _get_idxs_according_to_axis_sum(row_output, axis=0, percentage = percentage_column, greater_than = False)
            if len(idxs) > 0:
                thresh[:,idxs] = 0 # set idx meet target to value 0
            return thresh

        def sensitivity_process(column_output,sensitivity,neighborhood = 3):
            non_zero_columns = np.where(np.sum(column_output,axis=0) > 1)
            limit = np.floor(sensitivity * column_output.shape[1])
            f = np.insert(non_zero_columns ,0, [-limit] * neighborhood) # add one number to head
            f_ = np.append(f, [f[-1]+limit]*neighborhood).copy() # add one number to tail

            idxs = [ f_[i] for i in range(neighborhood,len(f_)-neighborhood) 
                    if (f_[i] - f_[i-neighborhood] > limit) and (f_[i+neighborhood] - f_[i] > limit) ]

            idxs = np.asarray(idxs).astype(np.uint8)
            if len(idxs) > 0:
                column_output[:,idxs] = 0 # set idx meet target to value 0   
            return column_output

        row_output,thresh = over_run_row_process(gray, percentage_row = percentage_row)
        column_output = over_run_column_process(row_output, thresh,percentage_column = percentage_column)
        sensitivity_output = sensitivity_process(column_output, sensitivity = sensitivity)
        return sensitivity_output



# if __name__ == '__main__':
    # os.chdir(r'D:\Github\FaurSound')

    # to_label_metaData_path = r'./cleaned_csv/cleaned_csv_2021-02-01_TO_LABEL.csv'
    # df = pd.read_csv(to_label_metaData_path, header = 0)
    # output_folder = r'D:\Solid\temp'
    
    # etiltWav.load_df_meta(df)

    # test_wav = r'data\from_e-tilt_noise_list_ppt\wav\60-over_running-shocks.wav'

    # obj = etiltWav(test_wav, n_fft=512, hop_length=128,get_infor_from_fn = False)
    # obj.cut_wav_in_second(left_cut=0.75, right_cut=0.75)

    # cv_output_folder = os.path.join(output_folder,'cv')
    # obj.stft_custom_spec_to_pic(output_folder=output_folder)
    # obj.save_stft_cv_pic(output_folder=cv_output_folder)

    # print(obj.df_meta.head(5))
    # print(obj.label)


