"""
Authors: Adam Li and Patrick Myers.

Version: 1.0
"""

import json
import os

import mne
import mne_bids
import numpy as np
import pandas as pd
from mne_bids import make_bids_basename

from eegio.base.utils.bids_helper import write_json


class BidsIO(object):
    """
    BIDS IO for EEG data base class.

    Attributes
    ----------
    bidsdir : Union[str, os.PathLike]
        The base directory for Bids data
    subject_id : str
        The unique identifier for the corresponding BidsPatient
    session_id : str
        The identifier for the BidsPatient session
    run_id : str
        The identifier for the snapshot recording
    kind : str
        The type of data in the snapshot recording

    """

    def __init__(
        self, bidsdir, subject_id, session_id, kind, run_id=None, datatype="edf"
    ):
        """
        Initialize a BidsIO Object.

        Should raise a ValueError if datatype is not supported

        Parameters
        ----------
        bidsdir :
        subject_id :
        session_id :
        kind :
        run_id :
        datatype :

        """
        self.rootdir = bidsdir
        self.subject_id = subject_id
        self.run_id = run_id
        self.session_id = session_id
        self.subjdir = os.path.join(
            bidsdir, make_bids_basename(subject=self.subject_id)
        )  # 'sub-' + self.subject_val)
        self.sessiondir = os.path.join(
            self.subjdir, make_bids_basename(session=self.session_id)
        )  # 'ses-' + self.session_id)
        self.datadir = os.path.join(self.sessiondir, kind)
        self.kind = kind
        self.datatype = datatype

        SUPPORTED_DATATYPES = ["edf", "fif", "nii", "nii.gz", "mgz"]
        if datatype not in SUPPORTED_DATATYPES:
            raise ValueError(f"Supported data file types are: {SUPPORTED_DATATYPES}")

    @property
    def bids_parameters(self):
        """
        Return a list of basic Bids parameters.

        Returns
        -------
        List of parameters

        """
        return [self.rootdir, self.subject_id, self.session_id, self.kind, self.run_id]

    @property
    def participantsjson_fpath(self):
        """
        Get the path of the participants json file.

        Returns
        -------
        Participants json path

        """
        return os.path.join(self.rootdir, "participants.json")

    @property
    def participantstsv_fpath(self):
        """
        Get the path of the participants tsv file.

        Returns
        -------
        Participants tsv path

        """
        return os.path.join(self.rootdir, "participants.tsv")

    @property
    def eventstsv_fpath(self):
        """
        Get the path of the events tsv file.

        Returns
        -------
        Events tsv path

        """
        return os.path.join(
            self.datadir,
            make_bids_basename(
                subject=self.subject_id,
                session=self.session_id,
                run=self.run_id,
                suffix="events.tsv",
            ),
        )

    @property
    def datafile_fpath(self):
        """
        Get the path of the data file.

        Returns
        -------
        Data file path

        """
        return os.path.join(
            self.datadir,
            make_bids_basename(
                subject=self.subject_id,
                session=self.session_id,
                run=self.run_id,
                suffix=f"{self.kind}.{self.datatype}",
            ),
        )

    @property
    def result_fpath(self):
        """
        Get the path of the results zipped numpy file.

        Returns
        -------
        Results path

        """
        return os.path.join(
            self.datadir,
            "results",
            make_bids_basename(
                subject=self.subject_id,
                session=self.session_id,
                run=self.run_id,
                suffix="results.npz",
            ),
        )

    @property
    def sidecarjson_fpath(self):
        """
        Get the path of the sidecar json file.

        Returns
        -------
        Sidecar json path

        """
        return os.path.join(
            self.datadir,
            make_bids_basename(
                subject=self.subject_id,
                session=self.session_id,
                run=self.run_id,
                suffix=f"{self.kind}.json",
            ),
        )

    @property
    def chanstsv_fpath(self):
        """
        Get the path of the channel tsv file.

        Returns
        -------
        Channel tsv path

        """
        return os.path.join(
            self.datadir,
            make_bids_basename(
                subject=self.subject_id,
                session=self.session_id,
                run=self.run_id,
                suffix="channels.tsv",
            ),
        )

    @property
    def scanstsv_fpath(self):
        """
        Get the path of the scans tsv file.

        Returns
        -------
        Scans tsv path

        """
        return os.path.join(
            self.sessiondir,
            make_bids_basename(
                subject=self.subject_id, session=self.session_id, suffix="scans.tsv"
            ),
        )

    """ relative params """

    @property
    def rel_participantsjson_fpath(self):
        """
        Get the path of the participants json file.

        Returns
        -------
        Participants json path

        """
        return os.path.join("participants.json")

    @property
    def rel_participantstsv_fpath(self):
        """
        Get the path of the participants tsv file.

        Returns
        -------
        Participants tsv path

        """
        return os.path.join("participants.tsv")

    @property
    def rel_eventstsv_fpath(self):
        """
        Get the path of the events tsv file.

        Returns
        -------
        Events tsv path

        """
        return os.path.join(
            make_bids_basename(subject=self.subject_id),
            make_bids_basename(session=self.session_id),
            self.kind,
            make_bids_basename(
                subject=self.subject_id,
                session=self.session_id,
                run=self.run_id,
                suffix="events.tsv",
            ),
        )

    @property
    def rel_datafile_fpath(self):
        """
        Get the path of the data file.

        Returns
        -------
        Data file path

        """
        return os.path.join(
            make_bids_basename(subject=self.subject_id),
            make_bids_basename(session=self.session_id),
            self.kind,
            make_bids_basename(
                subject=self.subject_id,
                session=self.session_id,
                run=self.run_id,
                suffix=f"{self.kind}.{self.datatype}",
            ),
        )

    @property
    def rel_result_fpath(self):
        """
        Get the path of the results zipped numpy file.

        Returns
        -------
        Results path

        """
        return os.path.join(
            make_bids_basename(subject=self.subject_id),
            make_bids_basename(session=self.session_id),
            "results",
            make_bids_basename(
                subject=self.subject_id,
                session=self.session_id,
                run=self.run_id,
                suffix="results.npz",
            ),
        )

    @property
    def rel_sidecarjson_fpath(self):
        """
        Get the path of the sidecar json file.

        Returns
        -------
        Sidecar json path

        """
        return os.path.join(
            make_bids_basename(subject=self.subject_id),
            make_bids_basename(session=self.session_id),
            self.kind,
            make_bids_basename(
                subject=self.subject_id,
                session=self.session_id,
                run=self.run_id,
                suffix=f"{self.kind}.json",
            ),
        )

    @property
    def rel_chanstsv_fpath(self):
        """
        Get the path of the channel tsv file.

        Returns
        -------
        Channel tsv path

        """
        return os.path.join(
            make_bids_basename(subject=self.subject_id),
            make_bids_basename(session=self.session_id),
            self.kind,
            make_bids_basename(
                subject=self.subject_id,
                session=self.session_id,
                run=self.run_id,
                suffix="channels.tsv",
            ),
        )

    @property
    def rel_scanstsv_fpath(self):
        """
        Get the path of the scans tsv file.

        Returns
        -------
        Scans tsv path

        """
        return os.path.join(
            make_bids_basename(subject=self.subject_id),
            make_bids_basename(session=self.session_id),
            make_bids_basename(
                subject=self.subject_id, session=self.session_id, suffix="scans.tsv"
            ),
        )


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
            raise OSError(
                f"{self.scanstsv_fpath} doesn't exist! Our parameters for loading are: "
                f"{self.bids_parameters}"
            )
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
            raise OSError(
                f"{self.chanstsv_fpath} doesn't exist! Our parameters for loading are: "
                f"{self.bids_parameters}"
            )
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
        participants = pd.read_csv(
            participants_fpath,
            index_col=False,
            header=0,
            delimiter="\t",
            dtype={"participant_id": str},
        )
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
        if "fif" in datafile_fpath:
            raw = mne.io.read_raw_fif(datafile_fpath)
        elif "edf" in datafile_fpath:
            raw = mne.io.read_raw_edf(datafile_fpath)

        raw = mne_bids.read_raw_bids(
            bids_fname=os.path.basename(self.datafile_fpath), bids_root=self.rootdir
        )
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
        metadata.to_csv(fpath, sep="\t", index=False)

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
        scans_df = scans_df.replace(np.nan, "n/a", regex=True)
        scans_df.to_csv(outfile, sep="\t", index=True)

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
        participants_df = participants_df.replace(np.nan, "n/a", regex=True)
        participants_df.to_csv(
            self.participantstsv_fpath,
            sep="\t",
            # index_label=True,
            index=False,
        )

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
