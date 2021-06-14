import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# from datetime import datetime, timedelta
from typing import List

import librosa
import librosa.display

# from adtk.data import validate_series
from itertools import count

from matplotlib.ticker import ScalarFormatter


class _wavNVH(object):

    """
    _wavNVH is high level class create based on wav file. it will contrain most common thing for a wav file
    for sound/noise analysis. the plan is we will create child-class for each mechatronics later on which will contrain their own property 
    """

    _ids = count(0)
    PIC_FORMAT = '.jpg'
    DPI = 100
    FIGSIZE = (12.5 , 10)

    def __init__(self, file_path: str, n_fft:int, hop_length:int) -> None:
        """init function will create common attribute for _wavNVH class.

        :param file_path: [file path for the wav file to processed on]
        :type file_path: str
        :param n_fft: [n_fft]
        :type n_fft: int
        :param hop_length: [hop_length]
        :type hop_length: int
        """
        self.id = next(self._ids)

        self._n_fft = n_fft
        self._hop_length = hop_length

        self.file_path = file_path
        self.fn = self._get_fn_from_file_path()

        self._n_mels = 128

        self.y = None
        self.sr = None
        self.duration = None
        self._load_audio()

        self.mel_spec = None
        # self._get_mel_spec()
        
        self.stft_spec = None
        self._get_stft_log_spec()

        # self.df_ts = None
        # self._get_df_for_adtk()
        # print(f'init _wavNVH done for {self.fn}')


    @staticmethod
    def check_wav_file_type(file_path:str, file_type:str='.wav') -> bool:
        """[check file's ext, see if match with file_type parameter]

        :param file_path: [file path or file name]
        :type file_path: [str]
        :param file_type: [str], defaults to '.wav'
        :type file_type: str, optional
        :return: [return true if matchs, false if not matched]
        :rtype: bool
        """
        _ , file_extension = os.path.splitext(file_path)
        if file_extension == file_type:
            return True
        else:
            return False


    def _get_fn_from_file_path(self) -> str:
        """get file name from a file path

        :return: [the file name]
        :rtype: str
        """
        fn = os.path.basename(self.file_path)

        if self.check_wav_file_type(self.file_path):
            return fn
        else:
            raise KeyError('input file are not wav file, please double check')

    
    def _update_duration(self) -> None:
        """this function update duration with current self.y
        """
        self.duration = len(self.y) / self.sr


    def _print_duration(self) -> None:
        """print duration of current y in second
        """
        
        print(f'length of wav is { "%.2f" % self.duration} second', end = '\n')


    def _load_audio(self, offset: float = None) -> None:
        """get y and sr from wav file, and save them into object related property
        during the load, sr are set to 22050 according to librosa default setting.

        :param offset: [time for start to read, in the audio file. in second ], defaults to None
        :type offset: float, optional
        """
        self.y, self.sr = librosa.load(self.file_path, sr=22050, offset=offset, mono=True)

        self._update_duration()


    def cut_wav_in_second(self, left_cut: float, right_cut: float, cut_margin:float = 0.1) -> None:
        """cut the audio left and right, according to second.

        :param left_cut: [left cut length, in second]
        :type left_cut: float
        :param right_cut: [right cut length, in second]
        :type right_cut: float
        :param cut_margin: [cut margin when checking cut length vs current duration]
        :type right_cut: float
        """
        if left_cut + right_cut >= self.duration * (1 - cut_margin):
            raise KeyError('wav duration is smaller than required cut length. please double check !')

        else:
            num_of_points_to_drop_left = int(left_cut * self.sr)
            num_of_points_to_drop_right = int(right_cut * self.sr)
            self.y = self.y[num_of_points_to_drop_left : - num_of_points_to_drop_right]
            self._update_after_cut_wav()
            
            # self._print_duration()  # no print for display progress bar in analySession
        

    def _update_after_cut_wav(self) -> None:
        """this function will update attributes linked with self.y after cut 
        """
        self._update_duration()
        # self._get_mel_spec()
        # self._get_df_for_adtk()
        self._get_stft_log_spec()


    # def _get_df_for_adtk(self) -> None:
    #     """this function create is to set up self.df_ts.
    #     self.df_ts is a df which have only 1 column = y(amplitude), and have datetime index start from '2021-01-01T00:00:10'
    #     goal is to use adtk package in the end.
    #     """

    #     df_wav = pd.DataFrame(self.y, columns=['amplitude'])
    #     date_start = datetime.fromisoformat('2021-01-01T00:00:10')
    #     date_list =  [date_start + timedelta(seconds= x*(1/self.sr)) for x in range(len(df_wav))]
    #     df_wav.index = date_list

    #     self.df_ts = validate_series(df_wav)

        
    def _get_mel_spec(self ) -> None:
        """this set mel spectrum for the object. according to given parameters.

        """
        
        mel_spectrogram = librosa.feature.melspectrogram(self.y, sr=self.sr, n_fft=self._n_fft,
                                                    hop_length= self._hop_length , n_mels=self._n_mels)
        self.mel_spec = librosa.power_to_db(mel_spectrogram)


    def _get_stft_log_spec(self) -> None:
        """this set stft log spectrum for the object. according to given parameters.

        """
        S_scale = librosa.stft(self.y, n_fft=self._n_fft, hop_length=self._hop_length)
        Y_scale = np.abs(S_scale) ** 2
        Y_log_scale = librosa.power_to_db(Y_scale)
        self.stft_spec = Y_log_scale
    

    def plot_mel_spec(self, vmin:int , vmax:int , figsize=FIGSIZE) -> None:
        """this function plot the mel spectrum

        :param vmin: [vmin]
        :type vmin: int
        :param vmax: [vmax]
        :type vmax: int
        :param figsize: [figsize], defaults to FIGSIZE
        :type figsize: tuple, optional
        """

        plt.figure(figsize = figsize)
        librosa.display.specshow(self.mel_spec, x_axis='s', y_axis='mel', sr=self.sr, hop_length=self._hop_length,
                            vmin=vmin, vmax=vmax)
        
        title , _ = os.path.splitext(self.fn)
        plt.title(title)

        plt.colorbar(format='%+2.f')
        plt.show()


    def plot_stft_spec(self, vmin:int, vmax:int , figsize=FIGSIZE) -> None:
        """this function plot the stft log spectrum

        :param vmin: [vmin]
        :type vmin: int
        :param vmax: [vmax]
        :type vmax: int
        :param figsize: [figsize], defaults to FIGSIZE
        :type figsize: tuple, optional
        """

        plt.figure(figsize = figsize)
        librosa.display.specshow(self.stft_spec, x_axis='s', y_axis='log', sr=self.sr, hop_length=self._hop_length,
                            vmin=vmin, vmax=vmax)

        title , _ = os.path.splitext(self.fn)
        plt.title(title)
        plt.colorbar(format='%+2.f')
        plt.show()


    def stft_spec_to_pic(self, save_folder:str, vmin:int,vmax:int , figsize=FIGSIZE) -> None:
        """
        this function save stft_spec to a pic

        :param save_folder: [folder_path to save the picture]
        :type save_folder: str
        :param vmin: [vmin]
        :type vmin: int
        :param vmax: [vmax]
        :type vmax: int
        :param figsize: [figsize], defaults to FIGSIZE
        :type figsize: tuple, optional
        """
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)

        plt.figure(figsize = figsize)
        librosa.display.specshow(self.stft_spec, x_axis='s', y_axis='log', sr=self.sr, hop_length=self._hop_length,
                            vmin=vmin, vmax=vmax)
        title, _ = os.path.splitext(self.fn)
        plt.axis('off')        
        plt.savefig(os.path.join(save_folder, title + self.PIC_FORMAT), bbox_inches='tight', pad_inches=0,transparent=False)

        print(f'spec saved as picture in {save_folder}, with name {title}.jpg')
        plt.close('all')


    def mel_spec_to_pic(self, save_folder:str, vmin:int , vmax:int ,figsize=(12, 10)) -> None:
        """
        this function save mel_spec to a pic

        :param save_folder: [folder_path to save the picture]
        :type save_folder: str
        :param vmin: [vmin]
        :type vmin: int
        :param vmax: [vmax]
        :type vmax: int
        :param figsize: [figsize], defaults to FIGSIZE
        :type figsize: tuple, optional
        """
        plt.figure(figsize = figsize)
        librosa.display.specshow(self.mel_spec, x_axis='s', y_axis='mel', sr=self.sr, hop_length=self._hop_length,
                            vmin=vmin, vmax=vmax)
        
        title, _ = os.path.splitext(self.fn)
        plt.axis('off')        
        plt.savefig(os.path.join(save_folder, title + '_mel.jpg'), bbox_inches='tight',pad_inches=0,transparent=False)
        print(f'spec saved as picture in {save_folder}, with name {title}.jpg')
        plt.close('all')


    def plot_wave(self, figsize=(16, 4), vlimit:float = 0.2) -> None:
        """plot wave in time domain.

        :param figsize: [size of figure], defaults to (16, 4)
        :type figsize: tuple, optional
        :param vlimit: [limit of y axis], defaults to 0.2
        :type vlimit: float, optional
        """
        fig, ax = plt.subplots(figsize=figsize)

        librosa.display.waveplot(y=self.y, sr=self.sr, alpha=0.6, ax=ax)
        ax.set(title='wave in time domain')
        ax.set_ylim(-vlimit,vlimit)
        

    def set_wav_length(self, target_length: float, cut_from_front:bool = False):
        """this function set wav length to target length, by cutting right part of the self.y

        :param target_length: [target length of wav, in second ]
        :type target_length: float
        :param cut_from_front: [if we cut from end or not ]
        :type cut_from_front: bool
        :raises KeyError: [raise error if target_length is longer than current duration]
        """
        
        if target_length > self.duration:
            raise KeyError('wav duration is small then target length')

        else:
            if cut_from_front:
                self.y = self.y[-int(target_length * self.sr):]
            else :
                self.y = self.y[:int(target_length * self.sr)]
        
        self._update_after_cut_wav()
        self._print_duration()
    

    @staticmethod
    def _coord_time(n, sr=22050, hop_length=128):
        """Get time coordinates from frames"""
        return librosa.core.frames_to_time(np.arange(n + 1), sr=sr, hop_length=hop_length)


    @staticmethod
    def _coord_fft_hz(n, sr=22050):
        """Get the frequencies for FFT bins"""
        n_fft = 2 * (n - 1)
        # The following code centers the FFT bins at their frequencies
        # and clips to the non-negative frequency range [0, nyquist]
        basis = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)
        fmax = basis[-1]
        basis -= 0.5 * (basis[1] - basis[0])
        basis = np.append(np.maximum(0, basis), [fmax])
        return basis