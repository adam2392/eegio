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

        self._add_original_scan_filename()
        self._convert_run_to_bids()
        self._modify_electrode_info()
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

    def _modify_electrode_info(self, status_dict: dict = None):
        """

        Parameters
        ----------
        status_dict : dict
            A possible dictionary created from an external data file

        Returns
        -------
        """
        self._add_notes_to_channels()
        channel_data = self.loader.load_channels_tsv()
        bad_channel_dict = self.find_bad_channels()
        for key, value in bad_channel_dict:
            channel_data = self.modify_electrode_status(channel_data, key, value)
        if status_dict is not None:
            for key, value in status_dict:
                channel_data = self.modify_electrode_status(channel_data, key, value)
            channel_types_dict = self.find_channel_types()
        for key, value in channel_types_dict:
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
            channel_dict[key] = value.uppper()
        return channel_dict

    def modify_electrode_type(self, channel_df: pd.DataFrame, chlabel: str, chtype: str):
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
                channel_df[ind]['type'] = chtype
        return channel_df

    def modify_electrode_status(self, channel_df: pd.DataFrame, chlabel: str, chdescription: str):
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
                channel_df[ind]['status'] = 'bad'
                channel_df[ind]['notes'] = chdescription
        return channel_df


    def _add_notes_to_channels(self):
        """
        Adds a column to the *channels.tsv file to allow for more detailed description of status beyond good/bad

        Returns
        -------

        """
        channels_data = self.loader.load_channels_tsv()
        colnames = channels_data.columns
        if 'Notes' not in colnames:
            notes = [''] * channels_data.shape[0]
            channels_data['Notes'] = notes
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
        sidecar_dict['EEGChannelCount'] = numbers['EEG']
        sidecar_dict['ECOGChannelCount'] = numbers['ECOG']
        sidecar_dict['SEEGChannelCount'] = numbers['SEEG']
        sidecar_dict['EOGChannelCount'] = numbers['EOG']
        sidecar_dict['ECGChannelCount'] = numbers['ECG']
        sidecar_dict['EMGChannelCount'] = numbers['EMG']
        sidecar_dict['MiscChannelCount'] = numbers['Misc']
        self.writer.write_sidecar_json()

    def _add_metadata(self, id, data):
        """
        Helper function to add patient metadata. Requires column ID and then data source to be passed in.

        Parameters
        ----------
        id :
        data :

        Returns
        -------

        """
        pass

    def _add_scandata(self, filename, acq_time, **scan_kwargs):
        pass
