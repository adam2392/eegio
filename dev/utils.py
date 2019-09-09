# -*- coding: utf-8 -*-
import contextlib
import io
import json
import ntpath
import os
import re

import numpy as np
import pandas as pd
import scipy.io as sio

from eegio.base.utils.data_structures_utils import NumpyEncoder
from eegio.base.utils.data_structures_utils import ensure_list


# def create_raw_struct(self, usebestmontage=False):
#     eegts = self.get_data()
#     chanlabels = self.chanlabels
#     samplerate = self.samplerate
#
#     # gets the best montage
#     best_montage = self.get_best_matching_montage(chanlabels)
#
#     # gets channel indices to keep
#     montage_chs = chanlabels
#     montage_data = eegts
#     if usebestmontage:
#         montage_inds = self.get_montage_channel_indices(
#             best_montage, chanlabels)
#         montage_chs = chanlabels[montage_inds]
#         other_inds = [idx for idx, ch in enumerate(
#             chanlabels) if ch not in montage_chs]
#         montage_data = eegts[montage_inds, :]
#
#         print("Removed these channels: ", chanlabels[other_inds])
#
#     mne_rawstruct = create_mne_topostruct_from_numpy(montage_data, montage_chs,
#                                                      samplerate, montage=best_montage)
#     return mne_rawstruct

# from eegio.base.utils import create_mne_topostruct_from_numpy
# @deprecated(version="0.1", reason="Applying outside now.")
# def apply_gaussian_kernel_smoothing(self, window_size):
#     from eegio.base.utils import PostProcess
#
#     mat = self.mat.copy()
#
#     # apply moving average filter to smooth out stuff
#     smoothed_mat = np.array([PostProcess.smooth_kernel(
#         x, window_len=window_size) for x in mat])
#
#     # realize that moving average messes up the ends of the window
#     # smoothed_mat = smoothed_mat[:, window_size // 2:-window_size // 2]
#
#     self.mat = smoothed_mat
#     return smoothed_mat
#
#
# @deprecated(version="0.1", reason="Applying outside now.")
# def timewarp_data(self, mat, desired_len):
#     def resample(seq, desired_len):
#         """
#         Function to resample an individual signal signal.
#
#         :param seq:
#         :param desired_len:
#         :return:
#         """
#         # downsample or upsample using Fourier method
#         newseq = scipy.signal.resample(seq, desired_len)
#         # or apply downsampling / upsampling based
#         return np.array(newseq)
#
#     if mat.ndim == 2:
#         newmat = np.zeros((mat.shape[0], desired_len))
#     elif mat.ndim == 3:
#         newmat = np.zeros((mat.shape[0], mat.shape[1], desired_len))
#     else:
#         raise ValueError(
#             "Matrix passed in can't have dimensions other then 2, or 3. Yours has: {}".format(mat.ndim))
#
#     for idx in range(mat.shape[0]):
#         seq = mat[idx, ...].squeeze()
#         newmat[idx, :] = resample(seq, desired_len)
#     return newmat
#
#     def get_cez_hemisphere(self):
#         hemisphere = []
#         if any('right' in x for x in self.cezlobe):
#             hemisphere.append('right')
#         if any('left' in x for x in self.cezlobe):
#             hemisphere.append('left')
#
#         if len(hemisphere) == 2:
#             warnings.warn(
#                 "Probably can't analyze patients with onsets in both hemisphere in lobes!")
#             print(self)
#
#         inds = []
#         for hemi in hemisphere:
#             for lobe in self.ch_groups.keys():
#                 if hemi in lobe:
#                     inds.extend(self.ch_groups[lobe])
#
#         return hemisphere, inds
#
#     def get_cez_quadrant(self):
#         quadrant_map = {
#             'right-front': ['right-temporal', 'right-frontal'],
#             'right-back': ['right-parietal', 'right-occipital'],
#             'left-front': ['left-temporal', 'left-frontal'],
#             'left-back': ['left-parietal', 'left-occipital']
#         }
#         quad1 = ['frontal', 'temporal']
#         quad2 = ['parietal', 'occipital']
#
#         print(self.cezlobe)
#         quadrant = []
#         for lobe in self.cezlobe:
#             if 'right' in lobe:
#                 if any(x in lobe for x in quad1):
#                     quadrant.extend(quadrant_map['right-front'])
#                 if any(x in lobe for x in quad2):
#                     quadrant.extend(quadrant_map['right-back'])
#             if 'left' in lobe:
#                 if any(x in lobe for x in quad1):
#                     quadrant.extend(quadrant_map['left-front'])
#                 if any(x in lobe for x in quad2):
#                     quadrant.extend(quadrant_map['left-back'])
#
#         if len(quadrant) == 2:
#             warnings.warn(
#                 "Probably can't analyze patients with onsets in all quadrants!")
#             print(self)
#
#         print(quadrant)
#
#         inds = []
#         for lobe in quadrant:
#             inds.extend(self.ch_groups[lobe])
#         return quadrant, inds
#
#     @deprecated(version="0.1", reason="Applying outside now.")
#     def apply_moving_avg_smoothing(self, window_size):
#         def movingaverage(interval, window_size):
#             window = np.ones(int(window_size)) / float(window_size)
#             return np.convolve(interval, window, 'same')
#
#         mat = self.mat.copy()
#
#         # apply moving average filter to smooth out stuff
#         smoothed_mat = np.array(
#             [movingaverage(x, window_size=window_size) for x in mat])
#
#         # realize that moving average messes up the ends of the window
#         smoothed_mat = smoothed_mat[:, window_size // 2:-window_size // 2]
#
#         self.mat = smoothed_mat
#         return smoothed_mat

# def load_contacts_regs(self, contact_regs, atlas=''):
#     self.contacts.load_contacts_regions(contact_regs)
#     self.atlas = atlas
#
# def load_chanxyz(self, chanxyz, referenx="T1MRI"):
#     """
#      Load in the channel's xyz coordinates.
#
#     :param chanxyz:
#     :param coordsystem:
#     :return:
#     """
#     if len(chanxyz) != self.ncontacts:
#         raise RuntimeError("In loading channels xyz, chanxyz needs to be"
#                            "of equal length as the number of contacts in dataset! "
#                            "There is a mismatch chanxyz={} vs "
#                            "dataset.ncontacts={}".format(
#             len(chanxyz), self.ncontacts
#         ))
#     self.contacts.load_contacts_xyz(chanxyz)
#     self.coordsystem = coordsystem


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


def writenpzfile(npzfilename, **kwargs):
    if not npzfilename.endswith('.npz'):
        npzfilename += '.npz'

    np.savez_compressed(npzfilename, **kwargs)


def reformat_rawdata(rawdata, samplepoints):
    samplepoints = samplepoints - samplepoints[0, 0]
    winsize = samplepoints[0, 1] - samplepoints[0, 0]
    formatted_data = np.zeros((len(samplepoints),
                               rawdata.shape[0],
                               winsize))

    formatted_data = []

    # loop through and format data into chunks of windows
    for i in range(len(samplepoints)):
        win = samplepoints[i, :].astype(int)
        data_win = rawdata[:, win[0]:win[1]]

        if data_win.shape[1] == winsize:
            # set window of data
            # formatted_data[i, ...] = data_win

            formatted_data.append(data_win)
    # formatted_data = np.moveaxis(formatted_data, 0, 2)
    # formatted_data = formatted_data.swapaxes(0, 1)
    # formatted_data = formatted_data.swapaxes(1, 2)
    return formatted_data


def create_data_masks(bad_channels, non_eeg_channels,
                      chanlabels, chanxyzlabels):
    # create mask from bad/noneeg channels
    badchannelsmask = bad_channels
    noneegchannelsmask = non_eeg_channels

    # create mask from raw recording data and structural data
    if len(chanxyzlabels) > 0:
        rawdatamask = np.array(
            [ch for ch in chanlabels if ch not in chanxyzlabels])
    else:
        rawdatamask = np.array([])
    if len(chanlabels) > 0:
        xyzdatamask = np.array(
            [ch for ch in chanxyzlabels if ch not in chanlabels])
    else:
        xyzdatamask = np.array([])

    return badchannelsmask, noneegchannelsmask, xyzdatamask, rawdatamask


def load_good_chans_inds(chanlabels, contact_regs=[], chanxyzlabels=[],
                         bad_channels=[], non_eeg_channels=[]):
    '''
    Function for only getting the "good channel indices" for
    data.

    It may be possible that some data elements are missing
    '''
    # get all the masks and apply them here to the TVB generated data
    # just sync up the raw data avail.
    rawdata_mask = np.ones(len(chanlabels), dtype=bool)
    _bad_channel_mask = np.ones(len(chanlabels), dtype=bool)
    _non_seeg_channels_mask = np.ones(len(chanlabels), dtype=bool)
    _gray_channels_mask = np.ones(len(chanlabels), dtype=bool)
    if len(chanxyzlabels) > 0:
        # sync up xyz and rawdata
        _, _, _, rawdatamask = create_data_masks(
            [], [], chanlabels, chanxyzlabels)
        # reject white matter contacts
        if len(contact_regs) > 0:
            white_matter_chans = np.array(
                [ch for idx, ch in enumerate(chanxyzlabels) if contact_regs[idx] == -1])
            _gray_channels_mask = np.array(
                [ch in white_matter_chans for ch in chanlabels], dtype=bool)

    # reject bad channels and non-seeg contacts
    _bad_channel_mask = np.array(
        [ch in bad_channels for ch in chanlabels], dtype=bool)
    _non_seeg_channels_mask = np.array(
        [ch in non_eeg_channels for ch in chanlabels], dtype=bool)
    rawdata_mask *= ~_bad_channel_mask
    rawdata_mask *= ~_non_seeg_channels_mask
    rawdata_mask *= ~_gray_channels_mask
    return rawdata_mask


def walk_up_folder(path, depth=1):
    _cur_depth = 1
    while _cur_depth < depth:
        path = os.path.dirname(path)
        _cur_depth += 1
    return path


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def compute_timepoints(numsignals, winsize, stepsize, samplerate):
    timepoints_ms = numsignals / samplerate

    # create array of indices of window start and end times
    timestarts = np.arange(0, timepoints_ms - winsize + 1, stepsize)
    timeends = np.arange(winsize - 1, timepoints_ms, stepsize)
    # create the timepoints array for entire data array
    timepoints = np.append(timestarts[:, np.newaxis],
                           timeends[:, np.newaxis], axis=1)

    return timepoints


def merge_metadata(metadata1, metadata2, overwrite=False):
    for key in metadata1.keys():
        if overwrite is False:
            if key not in metadata2.keys():
                metadata2[key] = metadata1[key]
        else:
            metadata2[key] = metadata1[key]
    return metadata2


def load_szinds(onsetind, offsetind, timepoints):
    timepoints = np.array(timepoints)
    # get the actual indices that occur within time windows
    if onsetind is not None:
        onsetwin = findtimewins(onsetind, timepoints)
    else:
        onsetwin = None
    if offsetind is not None:
        offsetwin = findtimewins(offsetind, timepoints)
    else:
        offsetwin = None
    return onsetwin, offsetwin


def compute_samplepoints(winsamps, stepsamps, numtimepoints):
    # Creates a [n,2] array that holds the sample range of each window that
    # is used to index the raw data for a sliding window analysis
    samplestarts = np.arange(
        0, numtimepoints - winsamps + 1., stepsamps).astype(int)
    sampleends = np.arange(winsamps, numtimepoints + 1, stepsamps).astype(int)

    # print(len(sampleends), len(samplestarts))
    samplepoints = np.append(samplestarts[:, np.newaxis],
                             sampleends[:, np.newaxis], axis=1)
    # print(samplepoints[len(sampleends)-1,:])
    return samplepoints


def map_to_win(index, samplepoints):
    if index is None:
        return None

    idx = (index >= samplepoints[:, 0]) * (index <= samplepoints[:, 1])
    indexind = np.where(idx)[0]
    if len(indexind) > 0:
        return indexind[0]
    else:
        return None


def map_samples_to_time(samplepoints, samplerate):
    return samplepoints / samplerate * 1000.


def findtimewins(times, timepoints):
    indices = []
    for time in ensure_list(times):
        if time == 0:
            indices.append(time)
        else:
            idx = (time >= timepoints[:, 0]) * (time <= timepoints[:, 1])
            timeind = np.where(idx)[0]
            if len(timeind) > 0:
                indices.append(timeind[0])
            else:
                indices.append(np.nan)
    return indices


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def getDirectories(directory):
    dirs = []
    for x in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, x)):
            dirs.append(x)
    return dirs


def writejsonfile(metadata, metafilename, overwrite=False):
    if os.path.exists(metafilename) and overwrite is False:
        raise OSError("Destination for meta json file exists! Please use option"
                      " overwrite=True to force overwriting.")
    with io.open(metafilename, 'w', encoding='utf8') as outfile:
        str_ = json.dumps(metadata,
                          indent=4, sort_keys=True, cls=NumpyEncoder,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(str_)


def loadjsonfile(metafilename):
    if not metafilename.endswith('.json'):
        metafilename += '.json'

    # encoding = "ascii", errors = "surrogateescape"
    try:
        with open(metafilename, mode='r', encoding='utf8', errors="ignore") as f:
            metadata = json.load(f)
        metadata = json.loads(metadata)
    except Exception as e:
        # print(e)
        with io.open(metafilename, errors="ignore",
                     encoding='utf8',
                     mode='r') as fp:
            json_str = fp.read()
        # print(json_str)
        try:
            metadata = json.loads(json_str)
        except:
            print(json_str)
        # with open(metafilename, mode='r', encoding='utf8', errors="ignore") as f:
        #     metadata = json.load(f)
        # metadata = json.loads(metadata)
    return metadata


def loadseegxyz(sensorsfile):
    '''
    This is just a wrapper function to retrieve the seeg coordinate data in a pd dataframe
    '''
    seeg_pd = pd.read_csv(
        sensorsfile,
        names=[
            'x',
            'y',
            'z'],
        delim_whitespace=True)
    return seeg_pd


def splitpatient(patient):
    stringtest = patient.find('seiz')

    if stringtest == -1:
        stringtest = patient.find('sz')
    if stringtest == -1:
        stringtest = patient.find('aw')
    if stringtest == -1:
        stringtest = patient.find('aslp')
    if stringtest == -1:
        stringtest = patient.find('_')
    if stringtest == -1:
        print(
            "Not sz, seiz, aslp, or aw! Please add additional naming possibilities, or tell data gatherers to rename datasets.")
    else:
        pat_id = patient[0:stringtest]
        seiz_id = patient[stringtest:]

        # remove any underscores
        pat_id = re.sub('_', '', pat_id)
        seiz_id = re.sub('_', '', seiz_id)

    return pat_id, seiz_id


def returnindices(pat_id, seiz_id=None):
    included_indices, onsetelecs, clinresult = returnnihindices(
        pat_id, seiz_id)
    if included_indices.size == 0:
        included_indices, onsetelecs, clinresult = returnlaindices(
            pat_id, seiz_id)
    if included_indices.size == 0:
        included_indices, onsetelecs, clinresult = returnummcindices(
            pat_id, seiz_id)
    if included_indices.size == 0:
        included_indices, onsetelecs, clinresult = returnjhuindices(
            pat_id, seiz_id)
    if included_indices.size == 0:
        included_indices, onsetelecs, clinresult = returntngindices(
            pat_id, seiz_id)
    return included_indices, onsetelecs, clinresult


def returntngindices(pat_id, seiz_id):
    included_indices = np.array([])
    onsetelecs = None
    clinresult = -1
    if pat_id == 'id001ac':
        # included_indices = np.concatenate((np.arange(0,4), np.arange(5,55),
        #                         np.arange(56,77), np.arange(78,80)))
        included_indices = np.array([0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                     15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                     32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48,
                                     49, 50, 51, 52, 53, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68,
                                     69, 70, 71, 72, 73, 74, 75, 76, 78, 79])
    elif pat_id == 'id002cj':
        # included_indices = np.array(np.arange(0,184))
        included_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8,
                                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                                     30, 31, 32, 33, 34, 35, 36, 37, 38,
                                     45, 46, 47, 48, 49, 50, 51, 52, 53,
                                     60, 61, 62, 63, 64, 65, 66, 67, 70, 71, 72, 73, 74, 75, 76, 85, 86, 87, 88, 89,
                                     90, 91, 92, 93, 100, 101, 102, 103, 104, 105,
                                     106, 107, 108, 115, 116, 117, 118, 119,
                                     120, 121, 122, 123, 129, 130, 131, 132, 133,
                                     134, 135, 136, 137,
                                     # np.arange(143, 156)
                                     143, 144, 145, 146, 147,
                                     148, 149, 150, 151, 157, 158, 159, 160, 161,
                                     162, 163, 164, 165, 171, 172, 173, 174, 175,
                                     176, 177, 178, 179, 180, 181, 182])
    elif pat_id == 'id003cm':
        included_indices = np.concatenate((np.arange(0, 13), np.arange(25, 37),
                                           np.arange(40, 50), np.arange(55, 69), np.arange(70, 79)))
    elif pat_id == 'id004cv':
        # removed OC'10, SC'5, CC'14/15
        included_indices = np.concatenate((np.arange(0, 23), np.arange(25, 39),
                                           np.arange(40, 59), np.arange(60, 110)))
    elif pat_id == 'id005et':
        included_indices = np.concatenate((np.arange(0, 39), np.arange(39, 47),
                                           np.arange(52, 62), np.arange(62, 87)))
    elif pat_id == 'id006fb':
        included_indices = np.concatenate((np.arange(10, 19), np.arange(40, 50),
                                           np.arange(115, 123)))
    elif pat_id == 'id008gc':
        included_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17,
                                     18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40,
                                     41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 61, 62, 63, 64, 65,
                                     71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 90, 92, 93,
                                     94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 110, 111])
    elif pat_id == 'id009il':
        included_indices = np.concatenate(
            (np.arange(0, 10), np.arange(10, 152)))
    elif pat_id == 'id010js':
        included_indices = np.concatenate((np.arange(0, 14),
                                           np.arange(
                                               15, 29), np.arange(
            30, 42), np.arange(
            43, 52),
            np.arange(53, 65), np.arange(66, 75), np.arange(76, 80),
            np.arange(81, 85), np.arange(86, 94), np.arange(95, 98),
            np.arange(99, 111),
            np.arange(112, 124)))
    elif pat_id == 'id011ml':
        included_indices = np.concatenate((np.arange(0, 18), np.arange(21, 68),
                                           np.arange(69, 82), np.arange(82, 125)))
    elif pat_id == 'id012pc':
        included_indices = np.concatenate((np.arange(0, 4), np.arange(9, 17),
                                           np.arange(
                                               18, 28), np.arange(
            31, 41), np.arange(
            44, 56),
            np.arange(57, 69), np.arange(70, 82), np.arange(83, 96),
            np.arange(97, 153)))
    elif pat_id == 'id013pg':
        included_indices = np.array([2, 3, 4, 5, 15, 18, 19, 20, 21, 23, 24,
                                     25, 30, 31, 32, 33, 34, 35, 36, 37, 38, 50, 51, 52, 53, 54, 55, 56,
                                     57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75,
                                     76, 77, 78])
    elif pat_id == 'id014rb':
        included_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                     14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                     33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                     51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
                                     68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
                                     85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
                                     102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
                                     116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
                                     130, 131, 132, 133, 135, 136, 140, 141, 142, 143, 144, 145, 146,
                                     147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
                                     160, 161, 162, 163, 164])

    elif pat_id == 'id015sf':
        included_indices = np.concatenate((np.arange(0, 37), np.arange(38, 77),
                                           np.arange(78, 121)))

    return included_indices, onsetelecs, clinresult


def returnnihindices(pat_id, seiz_id):
    included_indices = np.array([])
    onsetelecs = None
    clinresult = -1
    if pat_id == 'pt1':
        included_indices = np.concatenate((np.arange(0, 36), np.arange(41, 43),
                                           np.arange(45, 69), np.arange(71, 95)))
        onsetelecs = set(['ATT1', 'ATT2', 'AD1', 'AD2', 'AD3', 'AD4',
                          'PD1', 'PD2', 'PD3', 'PD4'])

        resectelecs = set(['ATT1', 'ATT2', 'ATT3', 'ATT4', 'ATT5', 'ATT6', 'ATT7', 'ATT8',
                           'AST1', 'AST2', 'AST3', 'AST4',
                           'PST1', 'PST2', 'PST3', 'PST4',
                           'AD1', 'AD2', 'AD3', 'AD4',
                           'PD1', 'PD2', 'PD3', 'PD4',
                           'PLT5', 'PLT6', 'SLT1'])
        clinresult = 1
    elif pat_id == 'pt2':
        # [1:14 16:19 21:25 27:37 43 44 47:74]
        included_indices = np.concatenate((np.arange(0, 14), np.arange(15, 19),
                                           np.arange(
                                               20, 25), np.arange(
            26, 37), np.arange(
            42, 44),
            np.arange(46, 74)))
        onsetelecs = set(['MST1', 'PST1', 'AST1', 'TT1'])
        resectelecs = set(['TT1', 'TT2', 'TT3', 'TT4', 'TT6', 'TT6',
                           'G1', 'G2', 'G3', 'G4', 'G9', 'G10', 'G11', 'G12', 'G18', 'G19',
                           'G20', 'G26', 'G27',
                           'AST1', 'AST2', 'AST3', 'AST4',
                           'MST1', 'MST2', 'MST3', 'MST4'])
        clinresult = 1
    elif pat_id == 'pt3':
        # [1:19 21:37 42:43 46:69 71:107]
        included_indices = np.concatenate((np.arange(0, 19), np.arange(20, 37),
                                           np.arange(41, 43), np.arange(45, 69), np.arange(70, 107)))
        onsetelecs = set(['SFP1', 'SFP2', 'SFP3',
                          'IFP1', 'IFP2', 'IFP3',
                          'MFP2', 'MFP3',
                          'OF1', 'OF2', 'OF3', 'OF4'])
        resectelecs = set(['FG1', 'FG2', 'FG9', 'FG10', 'FG17', 'FG18', 'FG25',
                           'SFP1', 'SFP2', 'SFP3', 'SFP4', 'SFP5', 'SFP6', 'SFP7', 'SFP8',
                           'MFP1', 'MFP2', 'MFP3', 'MFP4', 'MFP5', 'MFP6',
                           'IFP1', 'IFP2', 'IFP3', 'IFP4',
                           'OF3', 'OF4'])
        clinresult = 1

    elif pat_id == 'pt4':
        # [1:19 21:37 42:43 46:69 71:107]
        included_indices = np.concatenate((np.arange(0, 19), np.arange(20, 26),
                                           np.arange(28, 34)))
        onsetelecs = set([])
        resectelecs = set([])
        clinresult = -1

    elif pat_id == 'pt5':
        included_indices = np.concatenate((np.arange(0, 19), np.arange(20, 26),
                                           np.arange(28, 36)))

        onsetelecs = set([])
        resectelecs = set([])
        clinresult = -1

    elif pat_id == 'pt6':
        # [1:36 42:43 46 52:56 58:71 73:95]
        included_indices = np.concatenate((np.arange(0, 36), np.arange(41, 43),
                                           np.arange(45, 46), np.arange(51, 56), np.arange(57, 71), np.arange(72, 95)))
        onsetelecs = set(['LA1', 'LA2', 'LA3', 'LA4',
                          'LAH1', 'LAH2', 'LAH3', 'LAH4',
                          'LPH1', 'LPH2', 'LPH3', 'LPH4'])
        resectelecs = set(['LALT1', 'LALT2', 'LALT3', 'LALT4', 'LALT5', 'LALT6',
                           'LAST1', 'LAST2', 'LAST3', 'LAST4',
                           'LA1', 'LA2', 'LA3', 'LA4', 'LPST4',
                           'LAH1', 'LAH2', 'LAH3', 'LAH4',
                           'LPH1', 'LPH2'])
        clinresult = 2
    elif pat_id == 'pt7':
        # [1:17 19:35 37:38 41:62 67:109]
        included_indices = np.concatenate((np.arange(0, 17), np.arange(18, 35),
                                           np.arange(36, 38), np.arange(40, 62), np.arange(66, 109)))
        onsetelecs = set(['MFP1', 'LFP3',
                          'PT2', 'PT3', 'PT4', 'PT5',
                          'MT2', 'MT3',
                          'AT3', 'AT4',
                          'G29', 'G30', 'G39', 'G40', 'G45', 'G46'])
        resectelecs = set(['G28', 'G29', 'G30', 'G36', 'G37', 'G38', 'G39',
                           'G41', 'G44', 'G45', 'G46',
                           'LFP1', 'LFP2', 'LSF3', 'LSF4'])
        clinresult = 3
    elif pat_id == 'pt8':
        # [1:19 21 23 30:37 39:40 43:64 71:76]
        included_indices = np.concatenate((np.arange(0, 19), np.arange(20, 21),
                                           np.arange(
                                               22, 23), np.arange(
            29, 37), np.arange(
            38, 40),
            np.arange(42, 64), np.arange(70, 76)))
        onsetelecs = set(['G19', 'G23', 'G29', 'G30', 'G31',
                          'TO6', 'TO5',
                          'MST3', 'MST4',
                          'O8', 'O9'])
        resectelecs = set(['G22', 'G23', 'G27', 'G28', 'G29', 'G30', 'G31',
                           'MST2', 'MST3', 'MST4', 'PST2', 'PST3', 'PST4'])

        clinresult = 1
    elif pat_id == 'pt10':
        # [1:3 5:19 21:35 48:69]
        included_indices = np.concatenate((np.arange(0, 3), np.arange(4, 19),
                                           np.arange(20, 35), np.arange(47, 69)))
        onsetelecs = set(['TT1', 'TT2', 'TT4', 'TT6',
                          'MST1',
                          'AST2'])
        resectelecs = set(['G3', 'G4', 'G5', 'G6', 'G11', 'G12', 'G13', 'G14',
                           'TT1', 'TT2', 'TT3', 'TT4', 'TT5', 'TT6', 'AST1', 'AST2', 'AST3', 'AST4'])
        clinresult = 2
    elif pat_id == 'pt11':
        # [1:19 21:35 37 39 40 43:74 76:81 83:84]
        included_indices = np.concatenate((np.arange(0, 19), np.arange(20, 35),
                                           np.arange(
                                               36, 37), np.arange(
            38, 40), np.arange(
            42, 74),
            np.arange(75, 81), np.arange(82, 84)))
        onsetelecs = set(['RG29', 'RG30', 'RG31', 'RG37', 'RG38', 'RG39',
                          'RG44', 'RG45'])
        resectelecs = set(['RG4', 'RG5', 'RG6', 'RG7', 'RG12', 'RG13', 'RG14', 'RG15',
                           'RG21', 'RG22', 'RG23', 'RG29', 'RG30', 'RG31', 'RG37', 'RG38', 'RG39', 'RG45', 'RG46',
                           'RG47'])
        resectelecs = set(['RG4', 'RG5', 'RG6', 'RG7', 'RG12',
                           'RG13', 'RG14', 'RG15',
                           'RG21', 'RG22', 'RG23', 'RG29', 'RG30',
                           'RG31', 'RG37', 'RG38', 'RG39', 'RG45', 'RG46', 'RG47'])
        clinresult = 1
    elif pat_id == 'pt12':
        # [1:15 17:33 38:39 42:61]
        included_indices = np.concatenate((np.arange(0, 15), np.arange(16, 33),
                                           np.arange(37, 39), np.arange(41, 61)))
        onsetelecs = set(['AST1', 'AST2',
                          'TT2', 'TT3', 'TT4', 'TT5'])
        resectelecs = set(['G19', 'G20', 'G21', 'G22', 'G23', 'G27', 'G28', 'G29', 'G30', 'G31',
                           'TT1', 'TT2', 'TT3', 'TT4', 'TT5', 'TT6',
                           'AST1', 'AST2', 'AST3', 'AST4',
                           'MST1', 'MST2', 'MST3', 'MST4'])
        clinresult = 2
    elif pat_id == 'pt13':
        # [1:36 39:40 43:66 69:74 77 79:94 96:103 105:130]
        included_indices = np.concatenate((np.arange(0, 36), np.arange(38, 40),
                                           np.arange(
                                               42, 66), np.arange(
            68, 74), np.arange(
            76, 77),
            np.arange(78, 94), np.arange(95, 103), np.arange(104, 130)))
        onsetelecs = set(['G1', 'G2', 'G9', 'G10', 'G17', 'G18'])
        resectelecs = set(['G1', 'G2', 'G3', 'G4', 'G9', 'G10', 'G11',
                           'G17', 'G18', 'G19',
                           'AP2', 'AP3', 'AP4'])
        clinresult = 1
    elif pat_id == 'pt14':
        # [1:19 21:37 41:42 45:61 68:78]
        included_indices = np.concatenate((np.arange(0, 3), np.arange(6, 10),
                                           np.arange(
                                               11, 17), np.arange(
            18, 19), np.arange(
            20, 37),
            np.arange(40, 42), np.arange(44, 61), np.arange(67, 78)))
        onsetelecs = set(['MST1', 'MST2',
                          'TT1', 'TT2', 'TT3',
                          'AST1', 'AST2'])
        resectelecs = set(['TT1', 'TT2', 'TT3', 'AST1', 'AST2',
                           'MST1', 'MST2', 'PST1'])
        clinresult = 4
    elif pat_id == 'pt15':
        # [2:7 9:30 32:36 41:42 45:47 49:66 69 71:85];
        included_indices = np.concatenate((np.arange(1, 7), np.arange(8, 30),
                                           np.arange(
                                               31, 36), np.arange(
            40, 42), np.arange(
            44, 47),
            np.arange(48, 66), np.arange(68, 69), np.arange(70, 85)))
        onsetelecs = set(['TT1', 'TT2', 'TT3', 'TT4',
                          'MST1', 'MST2', 'AST1', 'AST2', 'AST3'])
        resectelecs = set(['G2', 'G3', 'G4', 'G5', 'G10', 'G11', 'G12', 'G13',
                           'TT1', 'TT2', 'TT3', 'TT4', 'TT5',
                           'AST1', 'AST2', 'AST3', 'AST4',
                           'MST1', 'MST2', 'MST3', 'MST4'])
        clinresult = 1
    elif pat_id == 'pt16':
        # [1:19 21:37 42:43 46:53]
        included_indices = np.concatenate((np.arange(0, 19), np.arange(20, 37),
                                           np.arange(41, 43), np.arange(45, 53)))
        onsetelecs = set(['TT1', 'TT2', 'TT3', 'TT4', 'TT5', 'TT6',
                          'AST1', 'AST2', 'AST3', 'AST4',
                          'MST3', 'MST4',
                          'G26', 'G27', 'G28', 'G18', 'G19', 'G20', 'OF4'])
        resectelecs = set(['G18', 'G19', 'G20', 'G26', 'G27', 'G28',
                           'G29', 'G30', 'TT1', 'TT2', 'TT3', 'TT4', 'TT5', 'TT6',
                           'AST1', 'AST2', 'AST3', 'AST4',
                           'MST1', 'MST2', 'MST3', 'MST4'
                           ])
        clinresult = 1
    elif pat_id == 'pt17':
        # [1:19 21:37 42:43 46:51]
        included_indices = np.concatenate((np.arange(0, 19), np.arange(20, 37),
                                           np.arange(41, 43), np.arange(45, 51)))
        onsetelecs = set(['TT1', 'TT2'])
        resectelecs = set(['G27', 'G28', 'G29', 'G30',
                           'TT', 'TT2', 'TT3', 'TT4', 'TT5', 'TT6',
                           'AST1', 'AST2', 'AST3', 'AST4',
                           'MST1', 'MST2', 'MST3', 'MST4'])
        clinresult = 1
    return included_indices, onsetelecs, clinresult


def returnlaindices(pat_id, seiz_id):
    included_indices = np.array([])
    onsetelecs = None
    clinresult = -1
    spreadelecs = None
    if pat_id == 'la01':
        # [1 3 7:8 11:13 17:19 22:26 32 34:35 37 42 50:55 58 ...
        #                 62:65 70:72 77:81 84:97 100:102 105:107 110:114 120:121 130:131];
        # onset_electrodes = {'Y''1', 'X''4', ...
        # 'T''5', 'T''6', 'O''1', 'O''2', 'B1', 'B2',...% rare onsets
        # }
        included_indices = np.concatenate((np.arange(0, 3), np.arange(6, 8), np.arange(10, 13),
                                           np.arange(
                                               16, 19), np.arange(
            21, 26), np.arange(
            31, 32),
            np.arange(
                                               33, 35), np.arange(
            36, 37), np.arange(
            41, 42),
            np.arange(
                                               49, 55), np.arange(
            57, 58), np.arange(
            61, 65),
            np.arange(
                                               69, 72), np.arange(
            76, 81), np.arange(
            83, 97),
            np.arange(
                                               99, 102), np.arange(
            104, 107), np.arange(
            109, 114),
            np.arange(119, 121), np.arange(129, 131)))
        onsetelecs = ["X'4", "T'5", "T'6", "O'1", "O'2", "B1", "B2"]
        spreadelecs = ["P1", "P2", 'P6', "X1", "X8", "X9", "E'2", "E'3"
                                                                  "T'1"]

        if seiz_id == 'inter2':
            included_indices = np.concatenate((np.arange(0, 1), np.arange(7, 16), np.arange(21, 28),
                                               np.arange(
                                                   33, 36), np.arange(
                39, 40), np.arange(
                42, 44), np.arange(
                46, 50),
                np.arange(
                                                   56, 58), np.arange(
                62, 65), np.arange(
                66, 68), np.arange(
                69, 75),
                np.arange(76, 83), np.arange(85, 89), np.arange(96, 103),
                np.arange(106, 109), np.arange(111, 115), np.arange(116, 117),
                np.arange(119, 123), np.arange(126, 127), np.arange(130, 134),
                np.arange(136, 137), np.arange(138, 144), np.arange(146, 153)))
        if seiz_id == 'ictal2':
            included_indices = np.concatenate((np.arange(0, 4), np.arange(6, 19), np.arange(20, 33),
                                               np.arange(
                                                   34, 37), np.arange(
                38, 40), np.arange(
                42, 98),
                np.arange(107, 136), np.arange(138, 158)))

            onsetelecs = ["Y'1"]
        clinresult = 1
    elif pat_id == 'la02':
        # [1:4 7 9 11:12 15:18 21:28 30:34 47 50:62 64:67 ...
        # 70:73 79:87 90 95:99]
        included_indices = np.concatenate((np.arange(0, 4), np.arange(6, 7), np.arange(8, 9),
                                           np.arange(
                                               10, 12), np.arange(
            14, 18), np.arange(
            20, 28),
            np.arange(
                                               29, 34), np.arange(
            46, 47), np.arange(
            49, 62),
            np.arange(
                                               63, 67), np.arange(
            69, 73), np.arange(
            78, 87),
            np.arange(89, 90), np.arange(94, 99)))
        onsetelecs = ["L'2", "L'3", "L'4"]
        clinresult = 1
    elif pat_id == 'la03':
        # [1:3 6:33 36:68 77:163]
        included_indices = np.concatenate((np.arange(0, 3), np.arange(5, 33),
                                           np.arange(35, 68), np.arange(76, 163)))
        onsetelecs = ["L7"]
        clinresult = 2
    elif pat_id == 'la04':
        # [1:4 9:13 15:17 22 24:32 44:47 52:58 60 63:64 ...
        # 67:70 72:74 77:84 88:91 94:96 98:101 109:111 114:116 121 123:129];
        included_indices = np.concatenate((np.arange(0, 4), np.arange(8, 13),
                                           np.arange(
                                               14, 17), np.arange(
            21, 22), np.arange(
            23, 32),
            np.arange(43, 47), np.arange(51, 58), np.arange(59, 60),
            np.arange(62, 64), np.arange(66, 70), np.arange(71, 74),
            np.arange(76, 84), np.arange(87, 91), np.arange(93, 96),
            np.arange(97, 101), np.arange(108, 111), np.arange(113, 116),
            np.arange(120, 121), np.arange(122, 129)))
        # FIRST ABLATION WAS A FAILURE
        onsetelecs = ["L'4", "G'1",  # 2ND RESECTION REMOVED ALL OF M' ELECTRODES
                      "M'1", "M'2", "M'3", "M'4", "M'5", "M'6", "M'7",
                      "M'8", "M'9", "M'10", "M'11", "M'12", "M'13", "M'14", "M'15", "M'16"]
        clinresult = 2
    elif pat_id == 'la05':
        # [2:4 7:15 21:39 42:82 85:89 96:101 103:114 116:121 ...
        # 126:145 147:152 154:157 160:161 165:180 182:191];
        included_indices = np.concatenate((np.arange(1, 4), np.arange(6, 15),
                                           np.arange(
                                               20, 39), np.arange(
            41, 82), np.arange(
            84, 89),
            np.arange(95, 101), np.arange(102, 114), np.arange(115, 121),
            np.arange(125, 145), np.arange(146, 152), np.arange(153, 157),
            np.arange(159, 161), np.arange(164, 180), np.arange(181, 191)))
        onsetelecs = ["T'1", "T'2", "D'1", "D'2"]
        clinresult = 1
    elif pat_id == 'la06':
        # [1:4 7:12 14:17 19 21:33 37 46:47 50:58 61:62 70:73 77:82 ...
        # 84:102 104:112 114:119];
        included_indices = np.concatenate((np.arange(0, 4), np.arange(6, 12),
                                           np.arange(
                                               13, 17), np.arange(
            18, 19), np.arange(
            20, 33),
            np.arange(36, 37), np.arange(45, 47), np.arange(49, 58),
            np.arange(60, 62), np.arange(69, 73), np.arange(76, 82),
            np.arange(83, 102), np.arange(103, 112), np.arange(113, 119)))
        onsetelecs = ["Q'3", "Q'4", "R'3", "R'4"]
        clinresult = 2
    elif pat_id == 'la07':
        # [1:18 22:23 25 34:37 44 48:51 54:55 57:69 65:66 68:78 ...
        # 82:83 85:93 96:107 114:120];
        included_indices = np.concatenate((np.arange(0, 4), np.arange(6, 18), np.arange(21, 23),
                                           np.arange(
                                               24, 25), np.arange(
            33, 37), np.arange(
            43, 44),
            np.arange(47, 51), np.arange(53, 55), np.arange(56, 69),
            np.arange(64, 66), np.arange(67, 78), np.arange(81, 83),
            np.arange(84, 93), np.arange(95, 107), np.arange(113, 120)))
        onsetelecs = ["T'1", "T'3", "R'8", "R'9"]
        clinresult = 1
    elif pat_id == 'la08':
        # [1:2 8:13 15:19 22 25 27:30 34:35 46:48 50:57 ...
        # 65:68 70:72 76:78 80:84 87:93 100:102 105:108 110:117 123:127 130:131 133:137 ...
        # 140:146]
        included_indices = np.concatenate((np.arange(0, 2), np.arange(7, 13),
                                           np.arange(
                                               14, 19), np.arange(
            21, 22), np.arange(
            24, 25),
            np.arange(26, 30), np.arange(33, 35), np.arange(45, 48),
            np.arange(49, 57), np.arange(64, 68), np.arange(69, 72),
            np.arange(75, 78), np.arange(79, 84), np.arange(86, 93),
            np.arange(99, 102), np.arange(104, 108), np.arange(109, 117),
            np.arange(122, 127), np.arange(129, 131), np.arange(132, 137),
            np.arange(139, 146)))
        onsetelecs = ["Q2"]
        clinresult = 2
    elif pat_id == 'la09':
        # [3:4 7:17 21:28 33:38 42:47 51:56 58:62 64:69 ...
        # 73:80 82:84 88:92 95:103 107:121 123 126:146 150:161 164:169 179:181 ...
        # 183:185 187:191]
        # 2/7/18 - got rid of F10 = looking at edf was super noisy
        included_indices = np.concatenate((np.arange(2, 3), np.arange(6, 17),
                                           np.arange(
                                               20, 28), np.arange(
            32, 38), np.arange(
            41, 47),
            np.arange(
                                               50, 56), np.arange(
            57, 62), np.arange(
            63, 66), np.arange(
            67, 69),
            np.arange(72, 80), np.arange(81, 84), np.arange(87, 92),
            np.arange(94, 103), np.arange(106, 121), np.arange(122, 123),
            np.arange(125, 146), np.arange(149, 161), np.arange(163, 169),
            np.arange(178, 181), np.arange(182, 185), np.arange(186, 191)))
        onsetelecs = ["X'1", "X'2", "X'3", "X'4", "U'1", "U'2"]
        if seiz_id == 'ictal2':
            included_indices = np.concatenate((np.arange(0, 4), np.arange(6, 19),
                                               np.arange(20, 39), np.arange(41, 189)))
            onsetelecs = ["P'1", "P'2"]

        clinresult = 2
    elif pat_id == 'la10':
        # [1:4 7:13 17:19 23:32 36:37 46:47 50 54:59 62:66 68:79 82:96 ...
        # 99:106 108:113 117:127 135:159 163:169 172:173 176:179 181:185];
        included_indices = np.concatenate((np.arange(0, 4), np.arange(6, 13),
                                           np.arange(
                                               16, 19), np.arange(
            22, 32), np.arange(
            35, 37),
            np.arange(45, 47), np.arange(49, 50), np.arange(53, 59),
            np.arange(61, 66), np.arange(67, 79), np.arange(81, 96),
            np.arange(98, 106), np.arange(107, 113), np.arange(116, 127),
            np.arange(134, 159), np.arange(162, 169), np.arange(171, 173),
            np.arange(175, 179), np.arange(180, 185)))
        onsetelecs = ["S1", "S2", "R2", "R3"]
        clinresult = 2
    elif pat_id == 'la11':
        # [3:4 7:16 22:30 33:39 42 44:49 53:62 64:87 91:100 ...
        # 102:117 120:127 131:140 142:191];
        included_indices = np.concatenate((np.arange(2, 4), np.arange(6, 16),
                                           np.arange(
                                               21, 30), np.arange(
            32, 39), np.arange(
            41, 42), np.arange(
            43, 49),
            np.arange(
                                               52, 62), np.arange(
            63, 87), np.arange(
            90, 100), np.arange(
            101, 117),
            np.arange(119, 127), np.arange(130, 140), np.arange(141, 191)))
        onsetelecs = ["D6", "Z10"]
        clinresult = 2
    elif pat_id == 'la12':
        included_indices = np.concatenate((np.arange(0, 4), np.arange(6, 15),
                                           np.arange(
                                               19, 23), np.arange(
            24, 31), np.arange(
            34, 36), np.arange(
            42, 44), np.arange(
            47, 48),
            np.arange(
                                               49, 59), np.arange(
            61, 66), np.arange(
            68, 86), np.arange(
            87, 90),
            np.arange(
                                               91, 100), np.arange(
            101, 119), np.arange(
            121, 129), np.arange(
            131, 134),
            np.arange(136, 150), np.arange(153, 154), np.arange(156, 161),
            np.arange(167, 178), np.arange(187, 191)))
        onsetelecs = ["S1", "S2", "R2", "R3"]
        clinresult = 3
    elif pat_id == 'la13':
        # [1:4 7:12 23:33 36:37 44:45 48:70 72:93]
        included_indices = np.concatenate((np.arange(0, 4), np.arange(6, 12),
                                           np.arange(
                                               22, 33), np.arange(
            35, 37), np.arange(
            43, 45),
            np.arange(47, 70), np.arange(71, 93)))
        onsetelecs = ["Y13", "Y14"]
        clinresult = 2
    elif pat_id == 'la15':
        # included_channels = [1:4 9:12 15:19 21:27 30:34 36:38 43:57 62:66 ...
        # 68:71 76:85 89:106 108:112 114:115 118:124 127:132 135:158 ...
        # 161:169 171:186]
        included_indices = np.concatenate((np.arange(0, 4), np.arange(8, 12),
                                           np.arange(
                                               14, 19), np.arange(
            20, 27), np.arange(
            29, 34),
            np.arange(35, 38), np.arange(42, 57), np.arange(61, 66),
            np.arange(67, 71), np.arange(75, 85), np.arange(88, 106),
            np.arange(107, 112), np.arange(113, 115), np.arange(117, 124),
            np.arange(126, 132), np.arange(134, 158), np.arange(160, 169),
            np.arange(170, 186)))

        if seiz_id == 'ictal':
            included_indices = np.concatenate((np.arange(0, 4), np.arange(6, 19),
                                               np.arange(
                                                   20, 39), np.arange(
                41, 95), np.arange(
                96, 112),
                np.arange(113, 132), np.arange(134, 187)))

        onsetelecs = ["R1", "R2", "R3"]
        clinresult = 4
    elif pat_id == 'la16':
        # [1:3 10:16 23:24 28 31:35 37:39 42:44 46:47 ...
        # 49:54 58:62 64:65 68:70 76:89 93:98 100:101 105:124 126 128:130 ...
        # 132:134 136:140 142:144 149:156 158:163 165:166 168:170 173:181
        # 183:189];
        included_indices = np.concatenate((np.arange(0, 3), np.arange(9, 16),
                                           np.arange(
                                               22, 24), np.arange(
            27, 28), np.arange(
            30, 35),
            np.arange(36, 39), np.arange(41, 44), np.arange(45, 47),
            np.arange(48, 54), np.arange(57, 62), np.arange(63, 65),
            np.arange(67, 70), np.arange(75, 89), np.arange(92, 98),
            np.arange(99, 101), np.arange(104, 124), np.arange(125, 126),
            np.arange(127, 130), np.arange(131, 134), np.arange(135, 140),
            np.arange(141, 144), np.arange(148, 156), np.arange(157, 163),
            np.arange(164, 166), np.arange(167, 170), np.arange(172, 181),
            np.arange(182, 189)))
        onsetelecs = ["Q7", "Q8"]
        clinresult = 4
    elif pat_id == 'la17':
        included_indices = np.concatenate((np.arange(0, 19), np.arange(20, 39),
                                           np.arange(41, 64)))
        onsetelecs = ["X'1", "Y'1"]
        clinresult = 4

    return included_indices, onsetelecs, clinresult


def returnummcindices(pat_id, seiz_id):
    included_indices = np.array([])
    onsetelecs = None
    clinresult = -1
    if pat_id == 'ummc001':
        # included_channels = [1:22 24:29 31:33 35:79 81:92];
        included_indices = np.concatenate((np.arange(0, 22), np.arange(23, 29), np.arange(30, 33),
                                           np.arange(34, 79), np.arange(80, 92)))
        onsetelecs = ["GP13", 'GP21', 'GP29']
        clinresult = 4
    elif pat_id == 'ummc002':
        # included_channels = [1:22 24:29 31:33 35:52];
        included_indices = np.concatenate((np.arange(0, 22), np.arange(23, 29), np.arange(30, 33),
                                           np.arange(34, 52)))
        onsetelecs = ['ANT1', 'ANT2', 'ANT3',
                      'MEST1', 'MEST2', 'MEST3', 'MEST4', 'GRID17', 'GRID25']
        # onsetelecs = ['ATT1', 'ATT2', 'ATT3',
        #         'MEST1', 'MEST2', 'MEST3', 'MEST4', 'GRID17', 'GRID25']
        clinresult = 1
    elif pat_id == 'ummc003':
        included_indices = np.concatenate((np.arange(0, 22), np.arange(23, 29), np.arange(30, 33),
                                           np.arange(34, 48)))
        onsetelecs = ['MEST4', 'MEST5', 'GRID4', 'GRID10', 'GRID12',
                      'GRID18', 'GRID19', 'GRID20', 'GRID26', 'GRID27']
        clinresult = 1
    elif pat_id == 'ummc004':
        included_indices = np.concatenate((np.arange(0, 22), np.arange(23, 29), np.arange(30, 33),
                                           np.arange(34, 49)))
        onsetelecs = ['AT1', 'GRID1', 'GRID9', 'GRID10', 'GRID17', 'GRID18']
        clinresult = 1
    elif pat_id == 'ummc005':
        included_indices = np.concatenate(
            (np.arange(0, 33), np.arange(34, 48)))
        onsetelecs = ['AT2', 'G17', 'G19', 'G25', 'G27', 'AT1', 'AT2', 'AT3', 'AT4',
                      'AT5', 'AT6']
        onsetelecs = ['AT1']
        # , 'GRID1', 'GRID9', 'GRID10', 'GRID17', 'GRID18']
        clinresult = 1
    elif pat_id == 'ummc005':
        included_indices = np.concatenate(
            (np.arange(0, 33), np.arange(34, 48)))
        onsetelecs = ['AT2', 'G17', 'G19', 'G25', 'G27']
        # , 'AT1', 'AT2', 'AT3', 'AT4','AT5', 'AT6']
        clinresult = 1
    elif pat_id == 'ummc006':
        included_indices = np.concatenate((np.arange(0, 22), np.arange(23, 26), np.arange(27, 29),
                                           np.arange(30, 33), np.arange(34, 56)))
        onsetelecs = [
            'MT2',
            'MT3',
            'MT4',
            'MES2',
            'MES3',
            'MES5',
            'MAT1',
            'MAT2']
        clinresult = 1
    elif pat_id == 'ummc007':
        included_indices = np.arange(0, 30)
        onsetelecs = ['LMES1', 'LMES2', 'LMES3', 'LMES4', 'LPT3', 'LANT4', 'LANT5',
                      'RMES1', 'RANT1', 'RANT2', 'RANT3', 'RANT4', 'RPT3', 'RPT4', 'RPT5']
        onsetelecs = [
            'MT1',
            'MT2',
            'MT3',
            'MT4',
            'MEST1',
            'MEST2',
            'MEST3',
            'MEST4',
            'MEST5']
        clinresult = 1
    elif pat_id == 'ummc007':
        included_indices = np.arange(0, 30)
        onsetelecs = ['LMES1', 'LMES2', 'LMES3', 'LMES4',
                      'RMES1', 'RANT1', 'RANT2', 'RANT3', 'RANT4']
        # 'LPT3','LANT4', 'LANT5',
        #  'RPT3', 'RPT4', 'RPT5']
        clinresult = 4
    elif pat_id == 'ummc008':
        included_indices = np.arange(0, 30)
        onsetelecs = ['GRID1', 'GRID2', 'GRID3', 'GRID4',
                      'GRID5', 'GRID11', 'GRID12', 'GRID13',
                      'GRID17', 'GRID18', 'GRID19', 'GRID20', 'GRID21',
                      'AT1', 'AT2', 'AT3', 'AT4',
                      'MT1', 'MT2', 'MT3', 'MT4']
        # 'GRID9',
        # 'GRID10',
        clinresult = 1
    elif pat_id == 'ummc009':
        included_indices = np.arange(0, 30)
        onsetelecs = ['G4', 'G5', 'G6', 'G7', 'G12', 'G14', 'PT1', 'AT1']
        clinresult = -1
    return included_indices, onsetelecs, clinresult


def returnjhuindices(pat_id, seiz_id):
    included_indices = np.array([])
    onsetelecs = None
    clinresult = -1
    if pat_id == 'jh103':
        included_indices = np.concatenate((np.arange(0, 4), np.arange(6, 12), np.arange(14, 23),
                                           np.arange(
                                               24, 33), np.arange(
            46, 63), np.arange(
            64, 66),
            np.arange(68, 71), np.arange(72, 110)))
        onsetelecs = ['RAD1', 'RAD2', 'RAD3', 'RAD4', 'RAD5',
                      'RAD6', 'RAD7', 'RAD8',
                      'RHD1', 'RHD2', 'RHD3', 'RHD4', 'RHD5',
                      'RHD6', 'RHD7', 'RHD8', 'RHD9',
                      'RTG40', 'RTG48']
        clinresult = 4
    elif pat_id == 'jh105':
        included_indices = np.concatenate((np.arange(0, 4), np.arange(6, 12), np.arange(13, 19),
                                           np.arange(
                                               20, 37), np.arange(
            41, 43), np.arange(
            45, 49),
            np.arange(50, 53), np.arange(54, 75), np.arange(77, 99)))
        onsetelecs = ['RPG4', 'RPG5', 'RPG6', 'RPG12', 'RPG13', 'RPG14', 'RPG20', 'RPG21',
                      'APD1', 'APD2', 'APD3', 'APD4', 'APD5', 'APD6', 'APD7', 'APD8',
                      'PPD1', 'PPD2', 'PPD3', 'PPD4', 'PPD5', 'PPD6', 'PPD7', 'PPD8',
                      'ASI3', 'PSI5', 'PSI6']
        clinresult = 1
    return included_indices, onsetelecs, clinresult


def clinregions(patient):
    ''' THE REAL CLINICALLY ANNOTATED AREAS '''
    # 001
    if 'id001' in patient:
        ezregions = ['ctx-rh-lateralorbitofrontal', 'ctx-rh-temporalpole']
        pzregions = [
            'ctx-rh-superiorfrontal',
            'ctx-rh-rostralmiddlefrontal',
            'ctx-lh-lateralorbitofrontal']
    if 'id002' in patient:
        ezregions = ['ctx-lh-lateraloccipital']
        pzregions = ['ctx-lh-inferiorparietal', 'ctx-lh-superiorparietal']
    if 'id003' in patient:
        ezregions = ['ctx-lh-insula']
        pzregions = ['Left-Putamen', 'ctx-lh-postcentral']
    if 'id004' in patient:
        ''' '''
        ezregions = [
            'ctx-lh-posteriorcingulate',
            'ctx-lh-caudalmiddlefrontal',
            'ctx-lh-superiorfrontal']
        pzregions = ['ctx-lh-precentral', 'ctx-lh-postcentral']
    if 'id005' in patient:
        ''' '''
        ezregions = ['ctx-lh-posteriorcingulate', 'ctx-lh-precuneus']
        pzregions = ['ctx-lh-postcentral', 'ctx-lh-superiorparietal']
    if 'id006' in patient:
        ''' '''
        ezregions = ['ctx-rh-precentral']
        pzregions = ['ctx-rh-postcentral', 'ctx-rh-superiorparietal']
    if 'id007' in patient:
        ''' '''
        ezregions = [
            'Right-Amygdala',
            'ctx-rh-temporalpole',
            'ctx-rh-lateralorbitofrontal']
        pzregions = ['Right-Hippocampus', 'ctx-rh-entorhinal', 'ctx-rh-medialorbitofrontal',
                     'ctx-rh-inferiortemporal', 'ctx-rh-temporalpole', 'ctx-rh-lateralorbitofrontal']  # 008
    if 'id008' in patient:
        ezregions = ['Right-Amygdala', 'Right-Hippocampus']
        pzregions = [
            'ctx-rh-superiortemporal',
            'ctx-rh-temporalpole',
            'ctx-rh-inferiortemporal',
            'ctx-rh-medialorbitofrontal',
            'ctx-rh-lateralorbitofrontal']
    if 'id009' in patient:
        ezregions = ['ctx-rh-lingual', 'ctx-rh-parahippocampal']
        pzregions = [
            'ctx-rh-lateraloccipital',
            'ctx-rh-fusiform',
            'ctx-rh-inferiorparietal']  # rlocc, rfug, ripc
    if 'id010' in patient:
        ezregions = [
            'ctx-rh-medialorbitofrontal',
            'ctx-rh-frontalpole',
            'ctx-rh-rostralmiddlefrontal',
            'ctx-rh-parsorbitalis']  # rmofc, rfp, rrmfg, rpor
        pzregions = ['ctx-rh-lateralorbitofrontal', 'ctx-rh-rostralmiddlefrontal',
                     'ctx-rh-superiorfrontal', 'ctx-rh-caudalmiddlefrontal']  # rlofc, rrmfc, rsfc, rcmfg
    if 'id011' in patient:
        ezregions = ['Right-Hippocampus', 'Right-Amygdala']  # rhi, ramg
        pzregions = ['Right-Thalamus-Proper', 'Right-Caudate', 'Right-Putamen',
                     'ctx-rh-insula', 'ctx-rh-entorhinal', 'ctx-rh-temporalpole']  # rth, rcd, rpu, rins, rentc, rtmp
    if 'id012' in patient:
        ezregions = [
            'Right-Hippocampus',
            'ctx-rh-fusiform',
            'ctx-rh-entorhinal',
            'ctx-rh-temporalpole']  # rhi, rfug, rentc, rtmp
        pzregions = ['ctx-lh-fusiform', 'ctx-rh-inferiorparietal', 'ctx-rh-inferiortemporal',
                     'ctx-rh-lateraloccipital', 'ctx-rh-parahippocampal', 'ctx-rh-precuneus',
                     'ctx-rh-supramarginal']  # lfug, ripc, ritg, rloc, rphig, rpcunc, rsmg
    # 013
    if 'id013' in patient:
        ezregions = ['ctx-rh-fusiform']
        pzregions = ['ctx-rh-inferiortemporal', 'Right-Hippocampus', 'Right-Amygdala',
                     'ctx-rh-middletemporal', 'ctx-rh-entorhinal']
    # 014
    if 'id014' in patient:
        ezregions = ['Left-Amygdala', 'Left-Hippocampus', 'ctx-lh-entorhinal', 'ctx-lh-fusiform',
                     'ctx-lh-temporalpole', 'ctx-rh-entorhinal']
        pzregions = ['ctx-lh-superiortemporal', 'ctx-lh-middletemporal', 'ctx-lh-inferiortemporal',
                     'ctx-lh-insula', 'ctx-lh-parahippocampal']
    if 'id015' in patient:
        ezregions = ['ctx-rh-lingual', 'ctx-rh-lateraloccipital', 'ctx-rh-cuneus',
                     'ctx-rh-parahippocampal', 'ctx-rh-superiorparietal', 'ctx-rh-fusiform',
                     'ctx-rh-pericalcarine']  # rlgg, rloc, rcun, rphig, rspc, rfug, rpc
        pzregions = [
            'ctx-rh-parahippocampal',
            'ctx-rh-superiorparietal',
            'ctx-rh-fusiform']  # rphig, rspc, rfug
    return ezregions, pzregions
