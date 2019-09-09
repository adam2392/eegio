from typing import List, Dict
import numpy as np

from eegio.loaders.loader import Loader
from eegio.writers.basewrite import BaseWrite


class Combiner(BaseWrite):
    def __init__(self, fpath):
        self.fpath = fpath

    def load_edf_files(self):
        for edffile in self.filelist:
            # initialize converter
            loader = Loader(datatype='ieeg')
            # load in the dataset and create metadata object
            edfconverter.load_file(filepath=edffile)

            # load in info data structure and edf annotated events
            edfconverter.extract_info_and_events(pat_id=patid)
            rawfif = edfconverter.convert_fif(bad_chans_list=[], save=False, replace=False)
            metadata = edfconverter.convert_metadata(patid, dataset_id, center, save=False)
            self.metadata_list.append(metadata)
            self.rawfif_list.append(rawfif)
            self.samplerate = rawfif.info['sfreq']

    def load_fif_files(self):
        pass

    def combine_datasets(self):
        print(self.rawfif_list)
        # combine and append the datasets
        for idx, dataset in enumerate(self.rawfif_list):
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
                               'events',
                               'onset',
                               'termination']

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
        #         print(events_list.shape)
        #         print(new_events.shape)
        for event in new_events:
            # print(event.shape)
            new_event = event
            new_event[0] = float(new_event[0]) + recording_length_seconds
            events_list = np.concatenate(
                (events_list, np.expand_dims(new_event, axis=0)), axis=0)

        #         print(recording_length_seconds)
        #         print(events_list)
        return events_list
