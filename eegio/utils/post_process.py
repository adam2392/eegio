import numpy as np
import scipy
import scipy.signal


class PostProcess(object):
    def __init__(self, mat=[]):
        self.mat = mat

    @staticmethod
    def movingaverage(interval, window_size):
        """
        Function to apply a moving average filter on a signal (interval)
        :param interval:
        :param window_size:
        :return:
        """
        window = np.ones(int(window_size)) / float(window_size)
        return np.convolve(interval, window, 'same')

    @staticmethod
    def resample(seq, desired_len):
        """
        Function to resample an individual signal signal.

        :param seq:
        :param desired_len:
        :return:
        """
        # downsample or upsample using Fourier method
        newseq = scipy.signal.resample(seq, desired_len)
        # or apply downsampling / upsampling based
        return np.array(newseq)

    @staticmethod
    def resample_mat(mat, desired_len):
        """
        Function to resample an entire matrix composed of signals x time.

        :param mat:
        :param desired_len:
        :return:
        """
        if mat.ndim == 2:
            newmat = np.zeros((mat.shape[0], desired_len))
        elif mat.ndim == 3:
            newmat = np.zeros((mat.shape[0], mat.shape[1], desired_len))

        for idx in range(mat.shape[0]):
            seq = mat[idx, ...].squeeze()
            newmat[idx, :] = PostProcess.resample(seq, desired_len)
        return newmat

    @staticmethod
    def time_warp_map(fragmat, startwin, stopwin, image_width):
        # warp the ictal region
        warped_ictal = PostProcess.resample_mat(
            fragmat[:, startwin:stopwin], image_width // 2)

        # get the preictal region
        preictal_mat = PostProcess.resample_mat(
            fragmat[:, 0:startwin], image_width//2)

        # get the postictal region
        postictal_mat = PostProcess.resample_mat(
            fragmat[:, stopwin:], image_width//3)

        # concatenate preictal and ictal warped region
        warped_mat = np.concatenate((preictal_mat, warped_ictal), axis=1)
        return warped_mat

    @staticmethod
    def format_map(fragmat, startwin, image_height, image_width):
        fragimage = np.zeros((image_height, image_width))

        # trim the dataset and resample before doing moving avg/metric
        print(startwin - image_width // 2, startwin + image_width // 2)
        if startwin - image_width // 2 < 0:
            fragmat = fragmat[:, 0:image_width]
        elif startwin + image_width // 2 > fragmat.shape[1]:
            fragmat = fragmat[:, -image_width:]
        else:
            fragmat = fragmat[:, startwin - image_width //
                              2:startwin + image_width // 2]

        fragimage[:fragmat.shape[0], :] = fragmat

        # get the trimmed final product
        return fragimage

    @staticmethod
    def format_orig_map(fragmat, startwin, image_width):
        # trim the dataset and resample before doing moving avg/metric
        print(startwin - image_width // 2, startwin + image_width // 2)

        # start window is smaller then a possible image width
        if startwin - image_width // 2 < 0:
            fragmat = fragmat[:, 0:image_width]

        elif startwin + image_width // 2 > fragmat.shape[1]:
            fragmat = fragmat[:, -image_width:]
        else:
            fragmat = fragmat[:, startwin - image_width //
                              2:startwin + image_width // 2]

        # get the trimmed final product
        return fragmat

    @staticmethod
    def smooth_kernel(x, window_len, window='hanning'):
        """smooth the data using a window with requested size.

        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal 
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.

        input:
            x: the input signal 
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.

        output:
            the smoothed signal

        example:

        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)

        see also: 

        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter

        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """

        if x.ndim != 1:
            raise(ValueError, "smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            raise(ValueError, "Input vector needs to be bigger than window size.")

        if window_len < 3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise(
                ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')
        return y


class ExtractRate():
    """
    Class to extract rate from a dataset given "spiking"
    """

    def __init__(self, ts):
        self.ts = ts

    def get_peaks(self, thresh=0.7):
        ts = self.ts.copy()
        ts[ts < thresh] = 0
        c_max_index = scipy.signal.argrelextrema(ts, np.greater, order=5)

        return c_max_index[0]

    def get_moving_rate(self):
        pass
