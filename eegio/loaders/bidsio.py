import copy
import json
import os
import pandas as pd
from datetime import datetime

from mne_bids import make_bids_basename


def write_dataset(dataset, output_dir, move=False):
    dataset = copy.deepcopy(dataset)
    dataset.set_path(os.path.abspath(output_dir))
    dataset.update(move=move)
    return dataset


def make_dirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


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
    def __init__(self, bidsdir, subject_id, session_id, run_id, kind):

        self.rootdir = bidsdir
        self.subjdir = os.path.join(bidsdir, make_bids_basename(subject=subject_id))
        self.sessiondir = os.path.join(self.subjdir, make_bids_basename(subject=subject_id,
                                                                        session=session_id))
        self.datadir = os.path.join(self.sessiondir, kind)
        self.subject_id = subject_id
        self.run_id = run_id
        self.session_id = session_id
        self.kind = kind

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
    def __init__(self, bidsdir, subject_id, session_id, run_id, kind="eeg"):
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


class BidsWriter():
    def __init__(self, bidsdir, subject_id, session_id, run_id, kind="eeg"):
        super(BidsWriter, self).__init__(bidsdir, subject_id, session_id, run_id, kind)

    def write_channels_tsv(self, metadata, fpath, overwrite=True):
        write_tsv(metadata, fpath, first_colum="")

    def write_sidecar_json(self):
        # TODO: Add original file name field
        pass

    def write_scans_tsv(self):
        pass

    def write_electrode_coords(self, coords_dict):
        pass

    def write_electrode_status(self, ):
        # TODO: Parse ECG, EMG, EOG and mark, but do not change status. Change status of only EEG with WM, or out
        pass
