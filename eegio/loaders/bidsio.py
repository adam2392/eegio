"""
Authors: Adam Li and Patrick Myers.

Version: 1.0
"""
import json
import os

import mne
import numpy as np
import pandas as pd
import mne_bids
from mne_bids import make_bids_basename
from eegio.base.objects.bids.basebids import BidsIO
from eegio.base.utils.bids_helper import write_json


class BidsLoader(BidsIO):
    """
    A class containing loading methods for various required BIDS files.

    Attributes
    ----------
    bidsdir : Union[str, os.PathLike]
        The base directory for bids data
    subject_id : str
        The unique identifier for the BidsPatient
    session_id : str
        The identifier for this BidsPatient session
    run_id : str
        The identifier for this snapshot recording
    kind : str
        The type of recording

    """

    def __init__(self, bidsdir, subject_id, session_id, run_id=None, kind="eeg"):
        """
        Initialize a BidsLoader object.

        Parameters
        ----------
        bidsdir :
        subject_id :
        session_id :
        run_id :
        kind :

        """
        super(BidsLoader, self).__init__(bidsdir, subject_id, session_id, kind, run_id)


    def load_sidecar_json(self):
        """
        Load in sidecar json associated with a recording file.

        Returns
        -------
        A dict representation of the json

        """
        sidecar_fpath = self.sidecarjson_fpath
        with open(sidecar_fpath) as file:
            data = json.load(file)
        return data

    def load_scans_tsv(self):
        """
        Load in the tsv file containing information about a patient's scans.

        Returns
        -------
        A pandas DataFrame object of the tsv

        """
        if not os.path.exists(self.scanstsv_fpath):
            raise OSError(f"{self.scanstsv_fpath} doesn't exist! Our parameters for loading are: "
                          f"{self.bids_parameters}")
        scans = pd.read_table(self.scanstsv_fpath)
        return scans

    def load_channels_tsv(self):
        """
        Load in the tsv file containing information about a patient's recording channels.

        Returns
        -------
        A pandas DataFrame object of the tsv

        """
        if not os.path.exists(self.chanstsv_fpath):
            raise OSError(f"{self.chanstsv_fpath} doesn't exist! Our parameters for loading are: "
                          f"{self.bids_parameters}")
        channels = pd.read_table(self.chanstsv_fpath, index_col=None)
        channels.dropna(inplace=True)
        return channels

    def load_participants_tsv(self):
        """
        Load in the tsv file containing information about all of the center's patients.

        Returns
        -------
        A pandas DataFrame object of the tsv

        """
        participants_fpath = self.participantstsv_fpath
        participants = pd.read_table(participants_fpath, dtype={'participant_id': str})
        return participants

    def load_participants_json(self):
        """
        Load in the json file containing fields for the center's patients.

        Returns
        -------
        A dict representation of the json

        """
        participants_fpath = self.participantsjson_fpath
        with open(participants_fpath) as file:
            data = json.load(file)
        return data

    def load_dataset(self):
        """
        Load in mne.io.Raw dataset based on the sessionid and runid of the dataset.

        TODO:
        1. Allow exotic return types in the form of easy to use data structures? or keep as MNE.io.Raw.
        TBD.

        Returns
        -------
        The mne.io.Raw dataset

        """
        datafile_fpath = self.datafile_fpath
        if 'fif' in datafile_fpath:
            raw = mne.io.read_raw_fif(datafile_fpath)
        elif 'edf' in datafile_fpath:
            raw = mne.io.read_raw_edf(datafile_fpath)

        raw = mne_bids.read_raw_bids(bids_fname=os.path.basename(self.datafile_fpath),
                                     bids_root=self.rootdir)
        return raw


class BidsWriter(BidsIO):
    """
    A class containing writing methods for various required BIDS files.

    Attributes
    ----------
    bidsdir : Union[str, os.PathLike]
        The base directory for bids data
    subject_id : str
        The unique identifier for the BidsPatient
    session_id : str
        The identifier for this BidsPatient session
    run_id : str
        The identifier for this snapshot recording
    kind : str
        The type of recording

    """

    def __init__(self, bidsdir, subject_id, session_id, run_id=None, kind="eeg"):
        """
        Initialize a BidsWriter Object.

        Parameters
        ----------
        bidsdir :
        subject_id :
        session_id :
        run_id :
        kind :

        """
        super(BidsWriter, self).__init__(bidsdir, subject_id, session_id, kind, run_id)

    def write_channels_tsv(self, metadata, fpath=None, overwrite=True):
        """
        Write the updated channel information to a file.

        Parameters
        ----------
        metadata : pandas.DataFrame
            The new channel information after modification
        fpath : os.PathLike
            The location to save this data
        overwrite : bool
            Whether to overwrite the existing channels tsv file

        """
        if fpath is None:
            fpath = self.chanstsv_fpath
        metadata.to_csv(fpath, sep='\t', index=False)

    def write_sidecar_json(self, sidecar_dict):
        """
        Write a passed dict object to the sidecar json location.

        Parameters
        ----------
        sidecar_dict : dict
            The data to write to the file.

        """
        outfile = self.sidecarjson_fpath
        write_json(outfile, sidecar_dict, overwrite=True)

    def write_scans_tsv(self, scans_df):
        """
        Write a passed DataFrame object to the scans tsv location.

        Parameters
        ----------
        scans_df : pandas.DataFrame
            The data to write to the file.

        """
        outfile = self.scanstsv_fpath
        scans_df = scans_df.replace(np.nan, 'n/a', regex=True)
        scans_df.to_csv(outfile, sep='\t', index=True)

    def write_electrode_coords(self, coords_dict, channels_df):
        """
        Write the coordinates for the electrodes in the channels tsv file.

        Parameters
        ----------
        coords_dict : Dict
            A dictionary with key=name, value=coordinate
        channels_df : pandas.DataFrame
            The original channel data to write to the file

        Returns
        -------
        The modified channel dataframe.

        """
        return channels_df

    def write_participants_tsv(self, participants_df):
        """
        Write a passed DataFrame object to the participants tsv location.

        Parameters
        ----------
        participants_df : pandas.DataFrame
            The updated information for the participants.tsv file

        """
        participants_df = participants_df.replace(np.nan, 'n/a', regex=True)
        participants_df.to_csv(self.participantstsv_fpath, sep='\t', index=True)

    def write_participants_json(self, participants_dict):
        """
        Write a passed dict object to the participants json location.

        Parameters
        ----------
        participants_dict : dict
            The updated dictionary for the participants.json file.

        """
        outfile = self.participantsjson_fpath
        write_json(outfile, participants_dict, overwrite=True)
