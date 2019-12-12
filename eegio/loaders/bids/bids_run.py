"""
Authors: Adam Li and Patrick Myers.

Version: 1.0
"""

import os
import pathlib
from typing import Dict

import mne
import mne_bids
from mne_bids.utils import _parse_bids_filename, _parse_ext

from eegio.loaders.bids.basebids import BaseBids
from eegio.loaders.bids.bidsio import BidsWriter, BidsLoader


class BidsRun(BaseBids):
    """
    The class for a specific eeg snapshot recording that follows BIDS format.

    The additional functionality added by this class allow for addition, deletion, and modification
    of metadata corresponding the the recording channel annotation and the sidecar json.

    Attributes
    ----------
    bids_root: Union[str, os.PathLike]
        Where this dataset is located (base directory)

    bids_fname : str
        The base filename of the BIDS compatible files. Typically, this can be
        generated using make_bids_basename.
        Example: `sub-01_ses-01_task-testing_acq-01_run-01`.
        This will write the following files in the correct subfolder of the
        output_path::

            sub-01_ses-01_task-testing_acq-01_run-01_meg.fif
            sub-01_ses-01_task-testing_acq-01_run-01_meg.json
            sub-01_ses-01_task-testing_acq-01_run-01_channels.tsv
            sub-01_ses-01_task-testing_acq-01_run-01_coordsystem.json

        and the following one if events_data is not None::

            sub-01_ses-01_task-testing_acq-01_run-01_events.tsv

        and add a line to the following files::

            participants.tsv
            scans.tsv

        Note that the modality 'meg' is automatically inferred from the raw
        object and extension '.fif' is copied from raw.filenames.

    verbose: bool
        verbosity

    """

    def __init__(self, bids_root, bids_fname: str, verbose: bool = True):
        super(BidsRun, self).__init__(bids_root=bids_root)

        # ensures just base path
        self.bids_fname = os.path.basename(bids_fname)

        # what is the modality -- meg, eeg, ieeg to read
        self.bids_basename = "_".join(bids_fname.split("_")[:-1])
        self.kind = bids_fname.split("_")[-1].split(".")[0]

        # extract BIDS parameters from the bids filename to be loaded/modified
        # gets: subjectid, sessionid, acquisition type, task name, runid, file extension
        params = _parse_bids_filename(self.bids_fname, verbose=verbose)
        self.subject_id, self.session_id = params["sub"], params["ses"]
        self.acquisition, self.task, self.run = (
            params["acq"],
            params["task"],
            params["run"],
        )
        _, self.ext = _parse_ext(self.bids_fname)

        self.result_fpath = None

        # instantiate a loader/writer
        self.loader = BidsLoader(
            bids_root=self.bids_root,
            bids_basename=self.bids_basename,
            kind=self.kind,
            datatype=self.ext,
        )
        self.writer = BidsWriter(
            bids_root=self.bids_root,
            bids_basename=self.bids_basename,
            kind=self.kind,
            datatype=self.ext,
        )

        if not os.path.exists(self.loader.datafile_fpath):
            raise RuntimeError(
                f"Bids dataset run does not exist at {self.loader.datafile_fpath}. "
                f"Please first create the dataset in BIDS format."
            )

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
        bids_dir = self.bids_root

        # copy over bids run to where it should be within the bids directory
        output_path = mne_bids.write_raw_bids(
            raw, bids_basename, output_path=bids_dir, overwrite=True, verbose=False
        )
        return output_path

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
        The sampling frequency in HZ

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
        """
        Add a field (i.e. key/value) to the sidecar json file.

        Parameters
        ----------
        key :
        value :

        Returns
        -------
        None
        """
        pass

    def append_channel_info(
        self, column_id: str, channel_id: str, value, channel_dict: Dict = None,
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
        if column_id in colnames:
            raise RuntimeError(
                f"Already added in the column: {column_id}. Call modify_channel_info to"
                f"modify."
            )

        if channel_dict is not None:
            new_data = []
            for key, value in channel_dict.items():
                new_data.append(value)
        else:
            new_data = ["N/A"] * channel_df.shape[0]
        channel_df[column_id] = new_data

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

    def delete_channel_info(self, column_id: str):
        """
        Remove column from the channels.tsv file.

        Parameters
        ----------
        column_id : (str)

        Returns
        -------
        None

        """
        channel_df = self.loader.load_channels_tsv()
        if column_id not in channel_df.columns:
            raise ValueError(
                f"Column id {column_id} not in the channels tsv file. Please pass a correct column or "
                f"add to the tsv file"
            )
        print("Inside delete: ", channel_df)
        channel_df.drop(columns=column_id, axis=1, inplace=True)
        print("Inside delete: ", channel_df)
        self.writer.write_channels_tsv(channel_df)

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

    # def modify_original_scan_filename(self, original_filename: str):
    #     """
    #     Add the passed original filename to the scans tsv file.
    #
    #     Parameters
    #     ----------
    #     original_filename :
    #         The path to the file from the source.
    #
    #     """
    #     scans_data = self.loader.load_scans_tsv()
    #     colnames = scans_data.columns
    #     if "original_filename" not in colnames:
    #         fnames = []
    #         for index, row in scans_data.iterrows():
    #             fnames.append("")
    #         scans_data["original_filename"] = fnames
    #     for index, row in scans_data.iterrows():
    #         if row["filename"] == self.fpath:
    #             scans_data["original_filename"][index] = original_filename
    #     self.writer.write_scans_tsv(scans_data)
