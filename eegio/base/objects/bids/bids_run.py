import os
from typing import Union, Dict

import pandas as pd
from mne_bids import make_bids_basename

from eegio.base.config import BAD_MARKERS
from eegio.base.objects.bids.basebids import BaseBids
from eegio.base.utils.bids_helper import BidsBuilder
from eegio.base.utils.scrubber import ChannelScrub
from eegio.loaders.bidsio import BidsIO, BidsWriter, BidsLoader


class BidsRun(BaseBids):
    """
    The class for a specific run of

    Attributes
    ----------

    Notes
    -----

    Examples
    --------
    """

    def __init__(
            self,
            subject_id: str,
            session_id: str,
            run_id: str,
            datadir: Union[str, os.PathLike] = None,
            modality: str = "ieeg",
            original_fileid: Union[str, os.PathLike] = None,
    ):
        super(BidsRun, self).__init__(datadir=datadir, modality=modality)
        self.subject_id = subject_id
        self.session_id = session_id
        self.run_id = run_id
        self.result_fpath = None
        self.original_fileid = original_fileid

        # instantiate a loader/writer
        '''
        Use these when accessing files directly. BidsIO should not be called.
        '''
        self.loader = BidsLoader(datadir, subject_id, session_id, run_id, modality)
        self.writer = BidsWriter(datadir, subject_id, session_id, run_id, modality)

        # upon loading a BidsRun, update the sidecar json's channel counts. This makes
        # it BIDS-compatible since these fields are required.
        self._update_sidecar_json_channelcounts()

    @property
    def fpath(self):
        return self.loader.datafile_fpath

    @property
    def sfreq(self):
        sidecar = self.loader.load_sidecar_json()
        return sidecar["SamplingFrequency"]

    @property
    def linefreq(self):
        sidecar = self.loader.load_sidecar_json()
        return sidecar["PowerLineFrequency"]

    @property
    def chs(self):
        """
        Gets the channel labels as a numpy array.

        Returns
        -------

        """
        chdf = self.loader.load_channels_tsv()
        return chdf["name"].to_numpy()

    @property
    def results_fpath(self):
        """

        Returns
        -------

        """
        return self.loader.result_fpath

    def load_data(self):
        """
        Wrapper function that loads in the dataset using BidsLoader.

        Returns
        -------

        """
        return self.loader.load_dataset()

    def get_run_metadata(self):
        """

        Returns
        -------

        """
        return self.loader.load_sidecar_json()

    def get_channels_metadata(self):
        """

        Returns
        -------

        """
        chdf = self.loader.load_channels_tsv()
        return chdf.to_dict()

    def get_channel_types(self):
        """
        Returns the channel types of this BidsRun as a dictionary with key as the channel name,
        and type as the value in the dictionary.

        TODO:
        1. make sure this works.

        Returns
        -------

        """
        chdf = self.loader.load_channels_tsv()
        return chdf[["name", "type"]].set_index("name")

    def get_channels_status(self):
        """
        Returns the channel status of this BidsRun as a dictionary with channel "name" as the key,
        and "status" as the value in the dictionary.

        TODO:
        1. make sure this works.

        Returns
        -------

        """
        chdf = self.loader.load_channels_tsv()
        return chdf[["name", "status"]].set_index("name")

    def modify_sidecar_json(self, key, value, overwrite=False):
        """
        Should modify sidecar json (represented as a dictionary) with a specified key and value. It will overwrite
        existing sidecar elements if already present. However, if overwrite is False, then it should raise a
        RunTimeError.


        Parameters
        ----------
        key :
        value :

        Returns
        -------

        """
        sidecar_dict = self.loader.load_sidecar_json()
        current_keys = sidecar_dict.keys()
        if key in current_keys and overwrite is False:
            raise RuntimeError(f'The passed key {key} already exists in the sidecar json file. Please pass overwrite'
                               f'as true if you wish to make the change')
        sidecar_dict[key] = value
        self.writer.write_sidecar_json(sidecar_dict)

    def modify_original_scan_filename(self, original_filename: str):
        """
        Helper function that would add the original scan filename. If it was not passed in correctly.

        Adds the original scan filename (from clinical source) to the scans tsv folder for this BidsPatient

        Returns
        -------

        """
        scans_data = self.loader.load_scans_tsv()
        colnames = scans_data.columns
        if 'original_filename' not in colnames:
            fnames = []
            for index, row in scans_data.iterrows():
                fnames.append('')
            scans_data['original_filename'] = fnames
        for index, row in scans_data.iterrows():
            if row['filename'] == self.fpath:
                row['original_filename'] = original_filename

    def append_channel_info(self, column_id: str, channel_id: str=None, value=None, channel_dict: Dict=None):
        """

        Parameters
        ----------
        column_id :
        channel_id :
        value :
        channel_dict :

        Returns
        -------

        """
        if value is not None and channel_dict is not None:
            raise TypeError("Passed in both value and channel dictionary. Only pass in one!")
        channel_df = self.loader.load_channels_tsv()
        colnames = channel_df.columns
        if column_id not in colnames:
            if channel_dict is not None:
                new_data = []
                for key, value in channel_dict.items():
                    new_data.append(value)
            else:
                new_data = [''] * channel_df.shape[0]
            channel_df[column_id] = new_data
        if value is not None:
            channel_df[column_id][channel_id] = value
        self.writer.write_channels_tsv(channel_df)

    def modify_channel_info(self, column_id: str, channel_id: str, value=None):
        """

        Parameters
        ----------
        column_id : str
            The column to modify. Must already exist in the tsv file
        channel_id : str
            The channel to modify. Must exist for the recording
        value :
            The value to add for the channel and column

        Returns
        -------

        """
        channel_df = self.loader.load_channels_tsv()
        colnames = channel_df.columns
        if column_id not in colnames:
            raise ValueError(f'Column id {column_id} not in the channels tsv file. Please pass a correct column or '
                             f'add to the tsv file')
        channels = channel_df['name'].tolist()
        if channel_id not in channels:
            raise LookupError(f'Passed channel {channel_id} not a valid channel for this recording.')
        channel_df[column_id][channel_id] = value
        self.writer.write_channels_tsv(channel_df)

    def find_bad_channels(self):
        """
        Finds bad electrodes based on hardcoded rules of the electrode name

        Returns
        -------
        A dictionary where the key is the electrode name and the value is a string description
        """
        channel_df = self.loader.load_channels_tsv()
        channel_scrub = ChannelScrub
        channel_names = channel_df['name']
        bad_channels = channel_scrub.look_for_bad_channels(channel_names)
        bad_channels_dict = {}
        for bad in bad_channels:
            bad_channels_dict[bad] = f"scrubbed channels containing markers {', '.join(BAD_MARKERS)}"
        return bad_channels_dict

    def find_channel_types(self):
        """
        Finds the actual type of the electrode based on a regular expression of the electrode name

        Returns
        -------
        A dictionary where the key is the electrode name and the value is the type
        """
        channel_df = self.loader.load_channels_tsv()
        channel_scrub = ChannelScrub
        channel_names = channel_df['name']
        channel_dict = channel_scrub.label_channel_types(channel_names)
        for key, value in channel_dict.items():
            channel_dict[key] = value.upper()
        return channel_dict

    @staticmethod
    def modify_electrode_type(channel_df: pd.DataFrame, chlabel: str, chtype: str):
        """
        Updates the run's channel information to be one of the ones in the following:
        https://github.com/bids-standard/bids-specification/blob/master/src/04-modality-specific-files/04-intracranial-electroencephalography.md

        Right now, we support EEG, ECOG, SEEG, ECG (i.e. EKG), EMG, EOG, MISC.

        All others are categorized as OTHER for now.

        TODO:
        1. add support for other types.

        Parameters
        ----------
        channel_df: pandas.DataFrame
            The current DataFrame containing electrode contact information
        chlabel: str
            The label of the electrode to modify
        chtype: str
            The type to change the electrode to

        Returns
        -------
        The modified DataFrame
        """
        for ind, row in channel_df.iterrows():
            if row['name'] == chlabel:
                channel_df['type'][ind] = chtype
        return channel_df

    @staticmethod
    def modify_electrode_status(channel_df: pd.DataFrame, chlabel: str, chdescription: str):
        """
        Updates the run's channel information by looking at the generated clinical electrode sheet annotations.
        Electrode status can either be "good", or "bad"

        Parameters
        ----------
        channel_df: pandas.DataFrame
            The current DataFrame containing electrode contact information
        chlabel: str
            The label of the electrode to modify
        chdescription: str
            The description to fill in the 'notes' column of the DataFrame


        Returns
        -------
        The modified DataFrame

        """
        for ind, row in channel_df.iterrows():
            if row['name'] == chlabel:
                channel_df['status'][ind] = 'bad'
                channel_df['notes'][ind] = chdescription
        return channel_df

    def _update_sidecar_json_channelcounts(self):
        """
        Helper function to update sidecar json for channel counts.
        Reads the current tsv file for the run's channels and updates the sidecar json with the channel type.

        Should be called everytime we modify anything in the sidecar json file.

        Returns
        -------

        """
        # load in channel tsv file as a dataframe
        channels_data = self.loader.load_channels_tsv()
        types = channels_data['type']

        # load in sidecar json as a dictionary
        sidecar_dict = self.loader.load_sidecar_json()
        numbers = types.value_counts()

        # go through each possible electrode type and update the count in sidecar json
        for chtype, count in zip(types, numbers):
            sidecar_dict[f"{chtype}ChannelCount"] = int(count)

        self.writer.write_sidecar_json(sidecar_dict)


    def _get_original_fileid(self):
        """

        Returns
        -------

        """
        scans_df = self.loader.load_scans_tsv()
        if self.run_id is None and self.original_fileid is None:
            raise ValueError('Must pass at least the run_id or original_fileid for an existing subject')
        if self.original_fileid is not None:
            return True
        elif self.run_id is not None:
            for ind, row in scans_df.iterrows():
                if self.run_id in row['filename']:
                    self.original_fileid = row['original_fileid']
                    return True
        if self.original_fileid is None:
            return False
