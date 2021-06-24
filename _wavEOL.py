import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
from itertools import count


class _wavEOL(object):

    """
    _wavNVH is high level class create based on wav file. it will contrain most common thing for a wav file
    for sound/noise analysis. the plan is we will create child-class for each mechatronics later on which will contrain their own property 
    """

    _ids = count(0)
    PIC_FORMAT = '.jpg'
    DPI = 100
    FIGSIZE = (12.5 , 10)

    def __init__(self, wav_file, n_fft:int, hop_length:int) -> None:
        """init function will create common attribute for _wavNVH class.

        :param wav_file: wav_file
        :type wav_file: file-like object
        :param n_fft: [n_fft]
        :type n_fft: int
        :param hop_length: [hop_length]
        :type hop_length: int
        """
        self.id = next(self._ids)
        self.wav_file = wav_file
        self._n_fft = n_fft
        self._hop_length = hop_length

        self.y = None
        self.sr = None
        self.duration = None
        self._load_audio()

        self.stft_spec = None
        self._get_stft_log_spec()

    
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
        self.y, self.sr = librosa.load(self.wav_file, sr=22050, offset=offset, mono=True)

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


    def _update_after_cut_wav(self) -> None:
        """this function will update attributes linked with self.y after cut 
        """
        self._update_duration()
        self._get_stft_log_spec()


    def _get_stft_log_spec(self) -> None:
        """this set stft log spectrum for the object. according to given parameters.

        """
        S_scale = librosa.stft(self.y, n_fft=self._n_fft, hop_length=self._hop_length)
        Y_scale = np.abs(S_scale) ** 2
        Y_log_scale = librosa.power_to_db(Y_scale)
        self.stft_spec = Y_log_scale
    

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