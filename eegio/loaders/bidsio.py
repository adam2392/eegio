import json
import os
from datetime import datetime

import pandas as pd
import mne
from mne_bids import make_bids_basename

from eegio.format.scrubber import ChannelScrub
from eegio.base.config import BAD_MARKERS

def write_tsv(data_dict, out_file, first_colum="id"):
    columns = get_all_sub_keys(data_dict)
    with open(out_file, "w") as opened_file:
        header = [first_colum] + list(columns)
        write_tsv_row(header, opened_file)
        for key, value in data_dict.items():
            column_values = list()
            for column in columns:
                if column in value:
                    column_values.append(data_value_to_string(value[column]))
                else:
                    column_values.append("")
            row = [key] + column_values
            write_tsv_row(row, opened_file)


def data_value_to_string(data):
    if isinstance(data, datetime):
        try:
            return datetime.strftime(data, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return "{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}".format(year=data.year,
                                                                                                  month=data.month,
                                                                                                  day=data.day,
                                                                                                  hour=data.hour,
                                                                                                  minute=data.minute,
                                                                                                  second=data.second)
    else:
        return str(data)


def write_json(data, out_file):
    with open(out_file, "w") as opened_file:
        json.dump(data, opened_file)


def write_tsv_row(row, opened_file):
    opened_file.write("\t".join(row) + "\n")


def get_all_sub_keys(data_dict):
    keys = set()
    for value in data_dict.values():
        for key in value.keys():
            keys.add(key)
    return keys


class BidsIO():
    """
    BIDS IO for EEG data base class.

    Parameters
    ----------
    bidsdir :
    subject_id :
    session_id :
    run_id :
    kind :

    """

    def __init__(self, bidsdir, subject_id, session_id, kind, run_id = None, datatype="edf"):
        self.rootdir = bidsdir
        self.subjdir = os.path.join(bidsdir, make_bids_basename(subject=subject_id))
        self.sessiondir = os.path.join(self.subjdir, make_bids_basename(subject=subject_id,
                                                                        session=session_id))
        self.datadir = os.path.join(self.sessiondir, kind)
        self.subject_id = subject_id
        self.run_id = run_id
        self.session_id = session_id
        self.kind = kind
        self.datatype = datatype

        SUPPORTED_DATATYPES = ["edf", "fif", "nii", "nii.gz", "mgz"]
        if datatype not in SUPPORTED_DATATYPES:
            raise ValueError(f"Supported data file types are: {SUPPORTED_DATATYPES}")

    @property
    def participantsjson_fpath(self):
        return os.path.join(self.datadir, "participants.json")

    @property
    def participantstsv_fpath(self):
        return os.path.join(self.datadir, "participants.tsv")

    @property
    def eventstsv_fpath(self):
        return os.path.join(self.datadir, make_bids_basename(subject=self.subject_id,
                                                             session=self.session_id,
                                                             run=self.run_id,
                                                             suffix="events.tsv"))

    @property
    def datafile_fpath(self):
        return os.path.join(self.datadir, make_bids_basename(subject=self.subject_id,
                                                             session=self.session_id,
                                                             run=self.run_id,
                                                             suffix=f"{self.kind}.{self.datatype}"))
    @property
    def result_fpath(self):
        return os.path.join(self.datadir, 'results', make_bids_basename(subject=self.subject_id,
                                                                        session=self.session_id,
                                                                        run=self.run_id,
                                                                        suffix="results.npz"))

    @property
    def sidecarjson_fpath(self):
        return os.path.join(self.datadir, make_bids_basename(subject=self.subject_id,
                                                             session=self.session_id,
                                                             run=self.run_id,
                                                             suffix=f"{self.kind}.json"))

    @property
    def chanstsv_fpath(self):
        return os.path.join(self.datadir, make_bids_basename(subject=self.subject_id,
                                                             session=self.session_id,
                                                             run=self.run_id,
                                                             suffix="channels.tsv"))

    @property
    def scanstsv_fpath(self):
        return os.path.join(self.sessiondir, make_bids_basename(subject=self.subject_id,
                                                                session=self.session_id,
                                                                suffix="scans.tsv"))


class BidsLoader(BidsIO):
    def __init__(self, bidsdir, subject_id, session_id, run_id = None, kind="eeg"):
        super(BidsLoader, self).__init__(bidsdir, subject_id, session_id, run_id, kind)

    def load_sidecar_json(self):
        """
        Helper method to load in the sidecar json associated with a recording file.

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
        Helper method to load in the tsv file containing information about a patient's scans

        Returns
        -------
        A pandas DataFrame object of the tsv
        """
        scans_fpath = self.scanstsv_fpath
        scans = pd.read_table(scans_fpath)
        return scans

    def load_channels_tsv(self):
        """
        Helper method to load in the tsv file containing information about a patient's recording channels

        Returns
        -------
        A pandas DataFrame object of the tsv
        """
        channels_fpath = self.chanstsv_fpath
        channels = pd.read_table(channels_fpath)
        return channels

    def load_participants_tsv(self):
        """
        Helper method to load in the tsv file containing information about all of the center's patients

        Returns
        A pandas DataFrame object of the tsv
        -------

        """
        participants_fpath = self.participantstsv_fpath
        participants = pd.read_table(participants_fpath)
        return participants

    def load_participants_json(self):
        """
        Helper method to load in the json file containing fields for the center's patients

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
        Helper method to load in mne.io.Raw dataset based on the sessionid and runid of the dataset

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
        #else:
        #    DatasetTypeError('Passed dataset must be of type')
        return raw


class BidsWriter(BidsIO):
    def __init__(self, bidsdir, subject_id, session_id, run_id=None, kind="eeg"):
        super(BidsWriter, self).__init__(bidsdir, subject_id, session_id, run_id, kind)

    def write_channels_tsv(self, metadata, fpath=self.chanstsv_fpath, overwrite=True):
        """
        Writes the updated channel information to a file

        Parameters
        ----------
        metadata : pandas.DataFrame
            The new channel information after modification
        fpath : os.PathLike
            The location to save this data
        overwrite : bool
            Whether to overwrite the existing channels tsv file
        Returns
        -------

        """
        write_tsv(metadata, fpath, first_colum="name")

    def write_sidecar_json(self):
        pass

    def write_scans_tsv(self):
        pass

    def write_electrode_coords(self, coords_dict, channels_df):
        return channels_df

    def write_electrode_status(self, status_dict, channels_df):
        """
        Updates the status of the electrodes based on hard-coded rules.

        Parameters
        ----------
        status_dict : dict
            The dictionary of electrode status information grabbed from the formatted clinical sheet
        channels_df : pandas.DataFrame
            The current BIDS layout of channel information containing default values

        Returns
        -------
        An updated version of channels_df with correct annotations for status
        """
        channel_scrub = ChannelScrub
        channel_names = channels_df['name']
        bad_channels = channel_scrub.look_for_bad_channels(channel_names)
        for key, value in bad_channels:
            for ind, row in channels_df.iterrows():
                if key == row['name']:
                    row['status'] = 'bad'
                    row['notes'] = f"scrubbed channels containing markers {', '.join(BAD_MARKERS)}"
        for key, value in status_dict.items():
            for ind, row in channels_df.iterrows():
                if key == row['name']:
                    row['status'] = value.status
                    row['notes'] = value.notes
        return channels_df

    def write_participants_tsv(self, new_df):
        """
        Updates the participants.tsv folder with new information

        Parameters
        ----------
        new_df : pandas.DataFrame
            The updated information for the participants.tsv file

        Returns
        -------

        """
        outfile = self.participantstsv_fpath
        write_tsv(new_df, outfile, 'participant_id')

    def write_participants_json(self, new_dict):
        """

        Parameters
        ----------
        new_dict : dict
            The updated dictionary for the participants.json file.

        Returns
        -------

        """
        outfile = self.participantsjson_fpath
        write_json(new_dict, outfile)

    def write_electrode_type(self, channels_df):
        """

        Parameters
        ----------
        channels_df : pandas.DataFrame
            The current information about the channels containing default type values.

        Returns
        -------
        An updated version of channels_df with the correct type information
        """
        channel_scrub = ChannelScrub
        channel_names = channels_df['name']
        channel_dict = channel_scrub.label_channel_types(channel_names)
        for key, value in channel_dict.items():
            for ind, row in channel_df.iterrows():
                if key == row['name']:
                    row['type'] = value.upper()
        return channels_df
