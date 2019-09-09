import numpy as np
import os

import pyedflib
import mne
import datetime
from eegio.format.dev.edfconversion.baseconverter import BaseConverter
from eegio.base.utils.utils import writejsonfile


class CombineEDF(BaseConverter):

    def __init__(self, fif_list, metadata_list):
        self.datasets = fif_list
        self.metadata_list = metadata_list

    def combine_dataset(self):
        # combine and append the datasets
        for idx, dataset in enumerate(self.datasets):
            if idx == 0:
                combined_dataset = dataset
            else:
                combined_dataset.append(dataset)

        print("Combined dataset: ", combined_dataset)

        self.samplerate = combined_dataset.info['sfreq']
        return combined_dataset

    def combine_metadata(self):
        for idx, metadata in enumerate(self.metadata_list):
            if idx == 0:
                combined_metadata = metadata
            else:
                combined_metadata = self._update_dict(
                    combined_metadata, metadata)
        return combined_metadata

    def _update_dict(self, master_dict, appendage_dict):
        TIME_DEPENDENT_KEYS = ['length_of_recording',
                               'events', 'onset', 'termination']

        prevlen = master_dict['length_of_recording']
        # samplerate = master_dict['samplerate']
        samplerate = self.samplerate
        prevsec = self._convert_sec(prevlen, samplerate)

        # print("Lengths of recordings: ", prevlen, samplerate, prevsec)
        for key in appendage_dict.keys():
            if key in TIME_DEPENDENT_KEYS:
                if key == 'length_of_recording':
                    master_dict[key] = appendage_dict[key] + prevlen
                elif key == 'onset' or key == 'termination':
                    master_dict[key] = appendage_dict[key] + prevsec
                elif key == 'events':
                    master_dict[key] = self._concat_events(master_dict[key],
                                                           appendage_dict[key],
                                                           prevsec)
            if key not in master_dict.keys():
                master_dict[key] = appendage_dict[key]

        return master_dict

    def _convert_sec(self, index, samplerate):
        return np.divide(index, samplerate)

    def _concat_events(self, events_list, new_events, recording_length_seconds):
        # print(events_list.shape)
        # print(new_events.shape)
        for event in new_events:
            # print(event.shape)
            new_event = event
            new_event[0] = float(new_event[0]) + recording_length_seconds
            events_list = np.concatenate(
                (events_list, np.expand_dims(new_event, axis=0)), axis=0)

        # print(recording_length_seconds)
        # print(events_list)
        return events_list

    def save_fif(self, fif_raw, dataset_metadata, datafilepath, replace=False):
        """
        Conversion function for the rawdata + metadata into a .fif file format with accompanying metadata .json
        object.

        rawdata + metadata_dict -> .fif + .json

        :param fpath:
        :param dataset_metadata:
        :param replace:
        :return:
        """
        # create a new information structure
        rawdata = fif_raw.get_data(return_times=False)
        assert rawdata.shape[0] == dataset_metadata['number_chans']

        fif_raw.save(datafilepath,
                     overwrite=replace,
                     verbose='ERROR')

        # create a filepath for the json object
        dataset_metadata['filename'] = os.path.basename(datafilepath)
        newmetafilepath = datafilepath.replace('_raw.fif', '.json')

        # save the formatted metadata json object
        writejsonfile(dataset_metadata, newmetafilepath, overwrite=replace)


# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 12:56:31 2018
@author: skjerns
Gist to save a mne.io.Raw object to an EDF file using pyEDFlib
(https://github.com/holgern/pyedflib)
Disclaimer:
    - Saving your data this way will result in slight 
      loss of precision (magnitude +-1e-09).
    - It is assumed that the data is presented in Volt (V), it will be internally converted to microvolt
    - Saving to BDF can be done by changing the file_type variable.
      Be aware that you also need to change the dmin and dmax to
      the corresponding minimum and maximum integer values of the
      file_type: e.g. BDF+ dmin, dmax =- [-8388608, 8388607]
"""


def write_edf(mne_raw, fname, events_list, picks=None, tmin=0, tmax=None, overwrite=False):
    """
    Saves the raw content of an MNE.io.Raw and its subclasses to
    a file using the EDF+ filetype
    pyEDFlib is used to save the raw contents of the RawArray to disk
    Parameters
    ----------
    mne_raw : mne.io.Raw
        An object with super class mne.io.Raw that contains the data
        to save
    fname : string
        File name of the new dataset. This has to be a new filename
        unless data have been preloaded. Filenames should end with .edf
    picks : array-like of int | None
        Indices of channels to include. If None all channels are kept.
    tmin : float | None
        Time in seconds of first sample to save. If None first sample
        is used.
    tmax : float | None
        Time in seconds of last sample to save. If None last sample
        is used.
    overwrite : bool
        If True, the destination file (if it exists) will be overwritten.
        If False (default), an error will be raised if the file exists.
    """
    if not issubclass(type(mne_raw), mne.io.BaseRaw):
        raise TypeError('Must be mne.io.Raw type')
    if not overwrite and os.path.exists(fname):
        raise OSError('File already exists. No overwrite.')
    # static settings
    file_type = pyedflib.FILETYPE_EDFPLUS
    sfreq = mne_raw.info['sfreq']
    date = datetime.now().strftime('%d %b %Y %H:%M:%S')
    first_sample = int(sfreq * tmin)
    last_sample = int(sfreq * tmax) if tmax is not None else None

    # convert data
    channels = mne_raw.get_data(picks,
                                start=first_sample,
                                stop=last_sample)

    # convert to microvolts to scale up precision
    channels *= 1e6

    # set conversion parameters
    dmin, dmax = [-32768, 32767]
    pmin, pmax = [channels.min(), channels.max()]
    n_channels = len(channels)

    # create channel from this
    try:
        f = pyedflib.EdfWriter(fname,
                               n_channels=n_channels,
                               file_type=file_type)

        channel_info = []
        data_list = []

        for i in range(n_channels):
            ch_dict = {'label': mne_raw.ch_names[i],
                       'dimension': 'uV',
                       'sample_rate': sfreq,
                       'physical_min': pmin,
                       'physical_max': pmax,
                       'digital_min': dmin,
                       'digital_max': dmax,
                       'transducer': '',
                       'prefilter': ''}

            channel_info.append(ch_dict)
            data_list.append(channels[i])

        f.setTechnician('mne-save-edf-adamli')
        f.setSignalHeaders(channel_info)
        for (onset_in_seconds, duration_in_seconds, description) in events_list:
            (onset_in_seconds, duration_in_seconds, description)
            f.writeAnnotation(onset_in_seconds,
                              duration_in_seconds, description)
        f.setStartdatetime(date)
        f.writeSamples(data_list)
    except Exception as e:
        print(e)
        return False
    finally:
        f.close()
    return True
