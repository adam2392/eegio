import os
from typing import Union, Dict
import pandas as pd

import warnings

from eegio.base.objects.bids.basebids import BaseBids
from eegio.base.utils.bids_helper import BidsBuilder
from eegio.loaders.bidsio import BidsIO, BidsWriter, BidsLoader

from eegio.base.utils.scrubber import ChannelScrub
from eegio.base.config import BAD_MARKERS

from mne_bids import make_bids_basename

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
            original_filename: str = None,
            original_subject_id: str = '',
    ):
        super(BidsRun, self).__init__(datadir=datadir, modality=modality)
        self.subject_id = subject_id
        self.session_id = session_id
        self.run_id = run_id
        self.original_filename = original_filename
        self.original_subject_id = original_subject_id
        self.result_fpath = None

        # instantiate a loader/writer
        self.loader = BidsLoader(datadir, subject_id, session_id, run_id, modality)
        self.writer = BidsWriter(datadir, subject_id, session_id, run_id, modality)

        self._convert_run_to_bids()
        self._add_original_scan_filename()
        self.modify_electrode_info()
        self._update_sidecar_json()


    @property
    def fpath(self):
        io = BidsIO(self.datadir, self.subject_id, self.session_id, self.modality, self.run_id, "edf")
        return io.datafile_fpath

    @property
    def results_fpath(self):
        io = BidsIO(self.datadir, self.subject_id, self.session_id, self.modality, self.run_id, "edf")
        return io.result_fpath

    def load_result(self):
        self.result_fpath = self.results_fpath

    def _convert_run_to_bids(self):
        bids_basename = make_bids_basename(subject=self.subject_id, session=self.session_id, run=self.run_id)
        builder = BidsBuilder()
        builder.convert_to_bids(edf_fpath=self.original_filename, bids_dir=self.datadir, bids_basename=bids_basename)
        # Possible that this breaks the participants.tsv. We need subject age and sex pulled from the edf file and added
        # to the participants.tsv file before running convert otherwise this will always add a new patient.
        # Option 2, just remove the newly added patient as shown below
        # builder.remove_added_patient(self.loader, self.writer, self.subject_id)

    def _add_original_scan_filename(self):
        """
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
                row['original_filename'] = self.original_filename

    def modify_electrode_info(self):
        """
        This method searches for automatic patterns in the electrode names to modify the channels.tsv file

        Returns
        -------
        """
        self._add_notes_to_channels()
        channel_data = self.loader.load_channels_tsv()
        bad_channel_dict = self.find_bad_channels()
        for key, value in bad_channel_dict.items():
            channel_data = self.modify_electrode_status(channel_data, key, value)
        channel_types_dict = self.find_channel_types()
        for key, value in channel_types_dict.items():
            channel_data = self.modify_electrode_type(channel_data, key, value)
        self.writer.write_channels_tsv(channel_data)

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

    def _add_notes_to_channels(self):
        """
        Adds a column to the *channels.tsv file to allow for more detailed description of status beyond good/bad

        Returns
        -------

        """
        channels_data = self.loader.load_channels_tsv()
        colnames = channels_data.columns
        if 'notes' not in colnames:
            notes = [''] * channels_data.shape[0]
            channels_data['notes'] = notes
        self.writer.write_channels_tsv(channels_data)

    def _update_sidecar_json(self):
        """
        Reads the current tsv file for the run's channels and updates the sidecar json with the channel type

        Returns
        -------

        """
        channels_data = self.loader.load_channels_tsv()
        types = channels_data['type']
        sidecar_dict = self.loader.load_sidecar_json()
        numbers = types.value_counts()
        try:
            sidecar_dict['EEGChannelCount'] = int(numbers['EEG'])
        except KeyError:
            sidecar_dict['EEGChannelCount'] = 0
        try:
            sidecar_dict['ECOGChannelCount'] = int(numbers['ECOG'])
        except KeyError:
            sidecar_dict['ECOGChannelCount'] = 0
        try:
            sidecar_dict['SEEGChannelCount'] = int(numbers['SEEG'])
        except KeyError:
            sidecar_dict['SEEGChannelCount'] = 0
        try:
            sidecar_dict['EOGChannelCount'] = int(numbers['EOG'])
        except KeyError:
            sidecar_dict['EOGChannelCount'] = 0
        try:
            sidecar_dict['ECGChannelCount'] = int(numbers['ECG'])
        except KeyError:
            sidecar_dict['ECGChannelCount'] = 0
        try:
            sidecar_dict['EMGChannelCount'] = int(numbers['EMG'])
        except KeyError:
            sidecar_dict['EMGChannelCount'] = 0
        try:
            sidecar_dict['MiscChannelCount'] = int(numbers['Misc'])
        except KeyError:
            sidecar_dict['MiscChannelCount'] = 0
        self.writer.write_sidecar_json(sidecar_dict)


