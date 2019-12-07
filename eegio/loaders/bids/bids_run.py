"""
Authors: Adam Li and Patrick Myers.

Version: 1.0
"""

import os
import pathlib
from typing import Union, Dict

import pandas as pd
import mne
import mne_bids
from eegio.base.config import BAD_MARKERS
from eegio.loaders.bids.basebids import BaseBids
from eegio.base.utils.scrubber import ChannelScrub
from eegio.loaders.bids.bidsio import BidsWriter, BidsLoader


class BidsRun(BaseBids):
    """
    The class for a specific eeg snapshot recording that follows BIDS format.

    The additional functionality added by this class allow for addition, deletion, and modification
    of metadata corresponding the the recording channel annotation and the sidecar json.

    Attributes
    ----------
    subject_id: str
        The unique identifier for the recorded subject.
    session_id:
        The identifier for this run's session.
    run_id:
        The identifier for the actual recording snapshot.
    datadir: Union[str, os.PathLike]
        Where this dataset is located (base directory)
    modality: str
        The method of data recording.
    original_fileid: str
        The source's file path for backwards annotation.

    """

    def __init__(
        self,
        subject_id: str,
        session_id: str,
        run_id: str,
        datadir: Union[str, os.PathLike] = None,
        modality: str = "ieeg",
        montage: str = None,
    ):
        """
        Initialize a BidsRun object.

        Parameters
        ----------
        subject_id :
        session_id :
        run_id :
        datadir :
        modality :

        """
        super(BidsRun, self).__init__(datadir=datadir, modality=modality)
        self.subject_id = subject_id
        self.session_id = session_id
        self.run_id = run_id
        self.result_fpath = None

        # instantiate a loader/writer
        self.loader = BidsLoader(datadir, subject_id, session_id, run_id, modality)
        self.writer = BidsWriter(datadir, subject_id, session_id, run_id, modality)

        if not os.path.exists(self.loader.datafile_fpath):
            raise RuntimeError(
                f"Bids dataset run does not exist at {self.loader.datafile_fpath}. "
                f"Please first create the dataset in BIDS format."
            )

        # upon loading a BidsRun, update the sidecar json's channel counts. This makes
        # it BIDS-compatible since these fields are required.
        self._update_sidecar_json_channelcounts()

    def _create_bidsrun(self, data_fpath):
        """
        Create a BIDS Run if it does not exist.

        Parameters
        ----------
        data_fpath : str
            A filepath to the dataset, and it will create a BIDS-like run using MNE_Bids.

        """
        if pathlib.Path(data_fpath).suffix == ".fif":
            raw = mne.io.read_raw_fif(data_fpath)
        elif pathlib.Path(data_fpath).suffix == ".edf":
            raw = mne.io.read_raw_edf(data_fpath)
        else:
            raise RuntimeError(
                "Can't automatically create bids run using this extension. "
            )
        bids_basename = self.loader.datafile_fpath
        bids_dir = self.datadir

        # copy over bids run to where it should be within the bids directory
        output_path = mne_bids.write_raw_bids(
            raw, bids_basename, output_path=bids_dir, overwrite=True, verbose=False
        )
        # return output_path

    @property
    def fpath(self):
        """
        Get the location of the run's data file in this directory.

        Returns
        -------
        The path to the data file

        """
        return self.loader.datafile_fpath

    @property
    def sfreq(self):
        """
        Get the sampling frequency used in this recording.

        Returns
        -------
        The sampling frequency

        """
        sidecar = self.loader.load_sidecar_json()
        return sidecar["SamplingFrequency"]

    @property
    def linefreq(self):
        """
        Get the power line frequency used in this recording (i.e. either 50 Hz or 60 Hz).

        Returns
        -------
        The power line frequency.

        """
        sidecar = self.loader.load_sidecar_json()
        return sidecar["PowerLineFrequency"]

    @property
    def chs(self):
        """
        Get the channel labels as a numpy array.

        Returns
        -------
        A list of channel names in the recording.

        """
        chdf = self.loader.load_channels_tsv()
        return chdf["name"].to_numpy()

    @property
    def results_fpath(self):
        """
        Get the path to the results directory.

        Returns
        -------
        The path of the results directory

        """
        return self.loader.result_fpath

    def load_data(self):
        """
        Load the dataset using BidsLoader.

        Returns
        -------
        The mne.io.Raw dataset

        """
        return self.loader.load_dataset()

    def get_run_metadata(self):
        """
        Load the sidecar json file as a dict object.

        Returns
        -------
        The dictionary of metadata

        """
        return self.loader.load_sidecar_json()

    def get_channels_metadata(self):
        """
        Load the channel metadata from the tsv file as a dict object.

        Returns
        -------
        The dictionary of channel annotations.

        """
        chdf = self.loader.load_channels_tsv()
        return chdf.to_dict()

    def get_channel_types(self):
        """
        Return the channel types of this BidsRun as a pandas DataFrame.

        The name is the index and the type is the column value. Can easily convert to
        key, value pair of a dict if desired.

        Returns
        -------
        A pandas DataFrame containing the name and type of each channel.

        """
        chdf = self.loader.load_channels_tsv()
        return chdf[["name", "type"]].set_index("name")

    def get_channels_status(self):
        """
        Return the channel status of this BidsRun as a pandas DataFrame.

        The name is the index and the status is the column value. Can easily convert to
        key, value pair of a dict if desired.

        Returns
        -------
        A pandas DataFrame containing the name and status of each channel.

        """
        chdf = self.loader.load_channels_tsv()
        return chdf[["name", "status"]].set_index("name")

    def delete_sidecar_field(self, key):
        """
        Remove a field from the sidecar json file for this run.

        Parameters
        ----------
        key : str
            Which field to remove.

        Returns
        -------
        The modified dictionary

        """
        sidecar_dict = self.loader.load_sidecar_json()
        sidecar_dict.pop(key, None)
        self.writer.write_sidecar_json(sidecar_dict)
        return sidecar_dict

    def modify_sidecar_field(self, key, value, overwrite=False):
        """
        Modify the sidecar json to have the new value in the passed field.

        It will overwrite existing sidecar elements if already present. However, if overwrite is False,
        then it should raise a RunTimeError.


        Parameters
        ----------
        key : str
            Which field to modify
        value : str
            The new value for the passed key

        Returns
        -------
        The modified dictionary

        """
        # always make keys capitalized in sidecar json
        key = key[0].upper() + key[1:]

        sidecar_dict = self.loader.load_sidecar_json()
        current_keys = sidecar_dict.keys()
        if key in current_keys and overwrite is False:
            raise RuntimeError(
                f"The passed key {key} already exists in the sidecar json file. Please pass overwrite"
                f"as true if you wish to make the change"
            )
        sidecar_dict[key] = value
        self.writer.write_sidecar_json(sidecar_dict)

    def add_sidecar_field(self, key, value):
        pass

    def append_channel_info(
        self,
        column_id: str,
        channel_id: str = None,
        value=None,
        channel_dict: Dict = None,
    ):
        """
        Add a new column to the channels tsv file.

        Parameters
        ----------
        column_id : str
            The name of the column to add
        channel_id : str
            The name of the singular channel to modify.
        value :
            The new value for the channel_id.
        channel_dict : Dict
            A dictionary of names and values for the new column

        """
        if value is not None and channel_dict is not None:
            raise TypeError(
                "Passed in both value and channel dictionary. Only pass in one!"
            )
        channel_df = self.loader.load_channels_tsv()
        colnames = channel_df.columns
        if column_id not in colnames:
            if channel_dict is not None:
                new_data = []
                for key, value in channel_dict.items():
                    new_data.append(value)
            else:
                new_data = [""] * channel_df.shape[0]
            channel_df[column_id] = new_data
        if value is not None:
            channel_df.loc[channel_df["name"] == channel_id, column_id] = value
        self.writer.write_channels_tsv(channel_df)

    def modify_channel_info(self, column_id: str, channel_id: str, value=None):
        """
        Modify some information about a single recording channel.

        Parameters
        ----------
        column_id : str
            The column to modify. Must already exist in the tsv file
        channel_id : str
            The channel to modify. Must exist for the recording
        value :
            The value to add for the channel and column

        """
        channel_df = self.loader.load_channels_tsv()
        if column_id not in channel_df.columns:
            raise ValueError(
                f"Column id {column_id} not in the channels tsv file. Please pass a correct column or "
                f"add to the tsv file"
            )
        if channel_id not in channel_df["name"].tolist():
            raise LookupError(
                f"Passed channel {channel_id} not a valid channel for this recording."
            )
        channel_df.loc[channel_df["name"] == channel_id, column_id] = value
        self.writer.write_channels_tsv(channel_df)

    def find_bad_channels(self):
        """
        Find bad electrodes based on hardcoded rules of the electrode name.

        Uses markers encoded in BAD_MARKERS (see config file)

        Returns
        -------
        A dictionary where the key is the electrode name and the value is a string description

        """
        channel_df = self.loader.load_channels_tsv()
        channel_scrub = ChannelScrub
        channel_names = channel_df["name"]
        bad_channels = channel_scrub.look_for_bad_channels(channel_names)
        bad_channels_dict = {}
        for bad in bad_channels:
            bad_channels_dict[
                bad
            ] = f"Scrubbed channels containing markers {', '.join(BAD_MARKERS)}"
        return bad_channels_dict

    def find_channel_types(self):
        """
        Find the actual type of the electrode based on a regular expression of the electrode name.

        Returns
        -------
        A dictionary where the key is the electrode name and the value is the type

        """
        channel_df = self.loader.load_channels_tsv()
        channel_scrub = ChannelScrub
        channel_names = channel_df["name"]
        channel_dict = channel_scrub.label_channel_types(channel_names)
        for key, value in channel_dict.items():
            channel_dict[key] = value.upper()
        return channel_dict

    def _update_sidecar_json_channelcounts(self):
        """
        Update sidecar json for channel counts.

        Reads the current tsv file for the run's channels and updates the sidecar json with the channel type.

        Should be called every time we modify anything in the sidecar json file.

        """
        # load in channel tsv file as a dataframe
        channels_data = self.loader.load_channels_tsv()
        types = channels_data["type"]

        # load in sidecar json as a dictionary
        sidecar_dict = self.loader.load_sidecar_json()
        numbers = types.value_counts()

        # go through each possible electrode type and update the count in sidecar json
        for chtype, count in zip(types, numbers):
            sidecar_dict[f"{chtype}ChannelCount"] = int(count)

        self.writer.write_sidecar_json(sidecar_dict)

        def modify_original_scan_filename(self, original_filename: str):
            """
            Add the passed original filename to the scans tsv file.

            Parameters
            ----------
            original_filename :
                The path to the file from the source.

            """
            scans_data = self.loader.load_scans_tsv()
            colnames = scans_data.columns
            if "original_filename" not in colnames:
                fnames = []
                for index, row in scans_data.iterrows():
                    fnames.append("")
                scans_data["original_filename"] = fnames
            for index, row in scans_data.iterrows():
                if row["filename"] == self.fpath:
                    scans_data["original_filename"][index] = original_filename
            self.writer.write_scans_tsv(scans_data)

    @staticmethod
    def modify_electrode_type(channel_df: pd.DataFrame, chlabel: str, chtype: str):
        """
        Modify a single recording channel's type.

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
        channel_df.loc([channel_df["name"] == chlabel])["type"] = chtype
        return channel_df

    # @staticmethod
    # def modify_electrode_status(
    #     channel_df: pd.DataFrame, chlabel: str, chdescription: str
    # ):
    #     """
    #     Update the run's channel information by looking at the generated clinical electrode sheet annotations.
    #
    #     Electrode status can either be "good", or "bad"
    #
    #     Parameters
    #     ----------
    #     channel_df: pandas.DataFrame
    #         The current DataFrame containing electrode contact information
    #     chlabel: str
    #         The label of the electrode to modify
    #     chdescription: str
    #         The description to fill in the 'notes' column of the DataFrame
    #
    #
    #     Returns
    #     -------
    #     The modified DataFrame
    #
    #     """
    #     # channel_df[channel_df["name"] == chlabel]["status"] = "bad"
    #     # channel_df[channel_df["name"] == chlabel]["notes"] = chdescription
    #     for ind, row in channel_df.iterrows():
    #         if row["name"] == chlabel:
    #             channel_df["status"][ind] = "bad"
    #             channel_df["notes"][ind] = chdescription
    #     return channel_df
