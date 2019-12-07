"""
Authors: Adam Li and Patrick Myers.

Version: 1.0
"""

import json
import os
from typing import Union

import mne
import mne_bids
import numpy as np
import pandas as pd
from mne_bids import make_bids_basename
from mne_bids import write_raw_bids, make_bids_folders
from mne_bids.utils import _parse_bids_filename


class BidsBuilder(object):
    """Suite of helper functions to build out a bids data directory. Makes folder directories and stuff"""

    @staticmethod
    def make_bids_folder_struct(datadir, subject, kind, session_id="one"):
        """
        Create the empty folder structure for a new subject according to the BIDS format.

        Session is assumed to be 'one' for now.

        Parameters
        ----------
        datadir :
        subject :
        kind :
        session_id :

        """
        if type(datadir) is not str:
            datadir = str(datadir)

        # create bids folders for this subject
        make_bids_folders(
            subject=subject,
            kind=kind,
            session=session_id,
            output_path=datadir,
            make_dir=True,
        )

    @staticmethod
    def make_preprocessed_folder(
        datadir, subject_id, session="one", preprocess_name="preprocessed"
    ):
        """
        Create the preproceesed data folder in the correct location if it does not already exist.

        Assumes session is 'one'.

        Parameters
        ----------
        datadir : Union[str, os.PathLike]
            The base directory for bids data
        subject_id : str
            The unique identifier for the BidsPatient
        session : str
            The identifier for this BidsPatient session
        preprocess_name : str
            The name of the preprocessed folder

        """
        basedir = os.path.join(
            datadir, make_bids_basename(subject=subject_id, session=session)
        )
        preprocessed_dir = os.path.join(basedir, preprocess_name)
        if os.path.exists(preprocessed_dir):
            return 1
        else:
            os.mkdir(preprocessed_dir)

    @staticmethod
    def make_results_folder(datadir, subject_id, session="one", result_name="results"):
        """
        Create the results folder in the correct location if it does not already exist.

        Assumes session is 'one'.

        Parameters
        ----------
        datadir :
        subject_id :
        session :
        result_name :

        """
        basedir = os.path.join(
            datadir, make_bids_basename(subject=subject_id, session=session)
        )
        results_dir = os.path.join(basedir, result_name)
        if os.path.exists(results_dir):
            return 1
        else:
            os.mkdir(results_dir)


class BidsConverter:
    @staticmethod
    def convert_to_bids(
        edf_fpath,
        bids_dir,
        bids_basename,
        excluded_contacts=None,
        eog_contacts=None,
        misc_contacts=None,
        overwrite=False,
        use_preprocess_pipeline=True,
    ):
        """
        Convert the passed edf file into the Bids format.

        Parameters
        ----------
        edf_fpath : Union[str, os.PathLike]
            The location the edf file.
        bids_dir : Union[str, os.PathLike]
            The base directory for newly created bids files.
        bids_basename : str
            The base name of the new data files
        excluded_contacts : list
            Contacts to be excluded from conversion
        eog_contacts : list
            Contacts to be annotated as EOG.
        misc_contacts : list
            Contacts to be annotated  as Misc.
        overwrite : bool
            Whether to overwrite an existing converted file.

        Returns
        -------
        The path to the new data file.

        """
        if excluded_contacts is None:
            excluded_contacts = ["-", ""]
        raw = mne.io.read_raw_edf(
            edf_fpath,
            preload=False,
            verbose="ERROR",
            exclude=excluded_contacts,
            eog=eog_contacts,
            misc=misc_contacts,
        )
        output_path = write_raw_bids(
            raw, bids_basename, output_path=bids_dir, overwrite=overwrite, verbose=False
        )
        return output_path

    @staticmethod
    def preprocess_into_fif(bids_fname, bids_dir, kind="eeg", overwrite=True):
        """
        Preprocess the edf file into fif format for easier manipulation.

        TODO:
        1. Determine if this is necessary. EDF files on one hand are hard to modify,
        but fif files have a lot of constraints.
        2. We should try to discuss w/ some folks that do this, or ask on gitter for mne_bids
        3. See: https://gist.github.com/skjerns/bc660ef59dca0dbd53f00ed38c42f6be/812cd1d4be24c0730db449ecc6eb0065da68ca51

        Parameters
        ----------
        bids_fname : Union[str, os.PathLike]
            The path to the bids format edf file
        bids_dir : Union[str, os.PathLike]
            The base directory for bids data
        kind : str
            The type of data contained in the edf file.
        overwrite : bool
            Whether to overwrite an existing converted file.

        Returns
        -------
        The path of the converted fif file

        """
        params = _parse_bids_filename(bids_fname, verbose=False)
        bids_basename = make_bids_basename(
            subject=params["sub"],
            session=params["ses"],
            run=params["run"],
            suffix="eeg.edf",
        )
        raw = mne_bids.read_raw_bids(bids_basename, bids_dir)

        # get events and convert to annotations
        events, events_id = mne.events_from_annotations(raw)
        onsets = events[:, 0] / raw.info["sfreq"]
        durations = np.zeros_like(onsets)  # assumes instantaneous events
        descriptions = events[:, 2]
        annotations = mne.annotations.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            orig_time=raw.info["meas_date"],
        )

        # add a bids run
        preprocessed_dir = os.path.join(
            bids_dir,
            make_bids_basename(subject=params["sub"]),
            make_bids_basename(session=params["ses"]),
            f"{kind}",
        )
        if not os.path.exists(preprocessed_dir):
            os.makedirs(preprocessed_dir)

        bids_basename = make_bids_basename(
            subject=params["sub"],
            session=params["ses"],
            run=params["run"],
            processing="fif",
            suffix="raw",
        )
        # save a fif copy and reload it
        raw.save(
            os.path.join(preprocessed_dir, bids_basename + ".fif"), overwrite=overwrite
        )
        raw = mne.io.read_raw_fif(
            os.path.join(preprocessed_dir, bids_basename + ".fif")
        )

        # convert into fif
        output_path = write_raw_bids(
            raw,
            bids_basename,
            output_path=bids_dir,
            overwrite=overwrite,
            # events_data=events,
            verbose=False,
        )
        return output_path


class BidsUtils:
    @staticmethod
    def add_field_to_participants_json(
        current_dict: dict, new_field: str, description: Union[str, dict]
    ):
        """
        Add a new field to the participants.json file.

        Parameters
        ----------
        current_dict : dict
            The dictionary to add to
        new_field : str
            The name of the new field
        description: Union[str, dict]
            The description of the new field
        Returns
        -------
        The modified dict object

        """
        new_dict = {new_field: {"Description": description}}

        current_dict.update(new_dict)
        return current_dict

    @staticmethod
    def add_subject_to_participants(subject_id: str, loader, writer):
        """
        Add a subject into the participants.tsv file.

        Parameters
        ----------
        subject_id : str
            The name of the subject to add to the participants file

        Returns
        -------
        The modified pandas DataFrame

        """
        participants_df = loader.load_participants_tsv()
        colnames = participants_df.columns
        new_df = pd.DataFrame(
            [[subject_id] + ["n/a"] * (len(colnames) - 1)], columns=colnames
        )
        if participants_df.empty:
            participants_df = new_df
        else:
            participants_df.append(new_df)
        participants_df = participants_df.set_index("participant_id")
        writer.write_participants_tsv(participants_df)
        return participants_df

    def create_channel_info(self, bidsrun):
        """
        Find and add necessary data to modify the channels file.

        This method searches for automatic patterns in the electrode names to modify the channels.tsv file

        Parameters
        ----------
        bidsrun: eegio.BidsRun
            The object containing information about this snapshot recording.

        """
        self.add_notes_to_channels(bidsrun)
        channel_data = bidsrun.loader.load_channels_tsv()
        bad_channel_dict = bidsrun.find_bad_channels()
        for key, value in bad_channel_dict.items():
            channel_data = bidsrun.modify_electrode_status(channel_data, key, value)
        channel_types_dict = bidsrun.find_channel_types()
        for key, value in channel_types_dict.items():
            channel_data = bidsrun.modify_electrode_type(channel_data, key, value)
        bidsrun.writer.write_channels_tsv(channel_data)

    @staticmethod
    def add_notes_to_channels(bidsrun):
        """
        Add a column to the *channels.tsv file to allow for more detailed description of status beyond good/bad.

        Parameters
        ----------
        bidsrun: eegio.BidsRun
            The object containing information about this snapshot recording.

        """
        channels_data = bidsrun.loader.load_channels_tsv()
        colnames = channels_data.columns
        if "notes" not in colnames:
            notes = [""] * channels_data.shape[0]
            channels_data["notes"] = notes
        bidsrun.writer.write_channels_tsv(channels_data)


def _to_tsv(data, fname):
    """
    Write an OrderedDict into a tsv file.

    Parameters
    ----------
    data : collections.OrderedDict
        Ordered dictionary containing data to be written to a tsv file.
    fname : str
        Path to the file being written.

    """
    n_rows = len(data[list(data.keys())[0]])
    output = _tsv_to_str(data, n_rows)

    with open(fname, "wb") as f:
        f.write(output.encode("utf-8"))


def _tsv_to_str(data, rows=5):
    """
    Return a string representation of the OrderedDict.

    Parameters
    ----------
    data : collections.OrderedDict
        OrderedDict to return string representation of.
    rows : int, optional
        Maximum number of rows of data to output.

    Returns
    -------
    String representation of the first `rows` lines of `data`.

    """
    col_names = list(data.keys())
    n_rows = len(data[col_names[0]])
    output = list()
    # write headings.
    output.append("\t".join(col_names))

    # write column data.
    max_rows = min(n_rows, rows)
    for idx in range(max_rows):
        row_data = list(str(data[key][idx]) for key in data)
        output.append("\t".join(row_data))

    return "\n".join(output)


def write_json(fname, dictionary, overwrite=False, verbose=False):
    """
    Write JSON to a file.

    Parameters
    ----------
    fname: Union[str, os.PathLike]
        The path fo the new json file
    dictionary: Dict
        The data to write to the json file
    overwrite: bool
        Whether to overwrite an existing json file with name fname.
    verbose: bool
        Whether to print the data to stdout

    """
    if os.path.exists(fname) and not overwrite:
        raise FileExistsError(
            '"%s" already exists. Please set '  # noqa: F821
            "overwrite to True." % fname
        )

    json_output = json.dumps(dictionary, indent=4)
    with open(fname, "w") as fid:
        fid.write(json_output)
        fid.write("\n")

    if verbose is True:
        print(os.linesep + "Writing '%s'..." % fname + os.linesep)
        print(json_output)


def write_tsv(fname, dictionary, overwrite=False, verbose=False):
    """
    Write an ordered dictionary to a .tsv file.

    Parameters
    ----------
    fname: Union[str, os.PathLike]
        The path fo the new json file
    dictionary: Dict
        The data to write to the json file
    overwrite: bool
        Whether to overwrite an existing json file with name fname.
    verbose: bool
        Whether to print the data to stdout

    """
    if os.path.exists(fname) and not overwrite:
        raise FileExistsError(
            '"%s" already exists. Please set '  # noqa: F821
            "overwrite to True." % fname
        )
    _to_tsv(dictionary, fname)

    if verbose:
        print(os.linesep + "Writing '%s'..." % fname + os.linesep)
        print(_tsv_to_str(dictionary))
