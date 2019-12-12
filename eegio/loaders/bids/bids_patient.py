"""
Authors: Adam Li and Patrick Myers.

Version: 1.0
"""
import os
import warnings
from collections import OrderedDict
from typing import Dict
from typing import Union

import bids
import mne_bids.utils
from mne_bids import make_bids_basename

from eegio.base.utils.bids_helper import BidsBuilder, BidsUtils
from eegio.loaders.bids.basebids import BaseBids
from eegio.loaders.bids.bidsio import BidsWriter, BidsLoader


class BidsPatient(BaseBids):
    """
    Class for patient data that inherits basic dataset functionality and follows Bids format.

    The additional components for this are related to the fact that each Patient can
    have multiple EEGTimeSeries, or Results related to it.

    Attributes
    ----------
    bids_fname: str
        Identifier for the session
    bids_root: Union[str, os.PathLike]
        The base directory for the Bids dataset that this BidsPatient belongs
    managing_user: str
        Managing user that is tied to this Patient. E.g. a clinical center, or a specific researcher,
        or specific doctor.
    modality: str
        The type of recording the BidsPatient underwent during this session.

    """

    def __init__(self, bids_root, subject_id: str, verbose):
        super(BidsPatient, self).__init__(bids_root=bids_root)

        # what is the modality -- meg, eeg, ieeg to read
        self.bids_basename = make_bids_basename(subject=subject_id)

        # extract BIDS parameters from the bids filename to be loaded/modified
        self.subject_id = subject_id

        # instantiate a loader/writer
        self.loader = BidsLoader(
            bids_root=self.bids_root, bids_basename=self.bids_basename,
        )
        self.writer = BidsWriter(
            bids_root=self.bids_root, bids_basename=self.bids_basename,
        )

        # run to cache and search to get all datasets available for this patient
        self._get_all_datasets()

        # run BIDS compatability checks
        self._bids_compatible_check()

    def __str__(self):
        """
        Print the string representation of BidsPatient.

        Returns
        -------
        str
            A description of the BidsPatient including the number of datasets.

        """
        return "{} Patient with {} Dataset Recordings ".format(
            self.subject_id, self.numdatasets
        )

    def _bids_compatible_check(self):
        """
        Check that the BIDS patient has the necessary folders and files.

        Checks existence of files:
        - participant tsv/json files
        - scans tsv file

        Returns
        -------
        None

        """
        validator = bids.BIDSValidator(index_associated=True)

        valid_check = []
        filepaths = {
            "participant_tsv_fpath": self.loader.rel_participantstsv_fpath,
            "participant_json_fpath": self.loader.rel_participantsjson_fpath,
            "scans_tsv_fpath": self.loader.rel_scanstsv_fpath,
        }

        for bids_name, filepath in filepaths.items():
            if not filepath.startswith("/"):
                filepath = "/" + filepath
            _check = validator.is_bids(filepath)
            valid_check.append(_check)
            if _check is False:
                raise RuntimeError(
                    f"Bids patient {self.subject_id} does not have valid filepaths associated at "
                    f"{bids_name}: {filepath}."
                )

        if any(x is False for x in valid_check):
            raise RuntimeError(
                f"Bids patient {self.subject_id} does not have valid filepaths associated."
            )

    def _loader(self, runid):
        """
        Create a BidsLoader for a specific runid.

        Parameters
        ----------
        runid : str
            Identifier for the desired recording

        """
        # instantiate a loader/writer
        self.loader = BidsLoader(bids_root=self.bids_root, bids_basename=None)

    def _write(self, runid):
        """
        Create a BidsWriter for a specific runid.

        Parameters
        ----------
        runid : str
            Identifier for the desired recording

        """
        self.writer = BidsWriter(bids_root=self.bids_root, bids_basename=None)

    def _create_derivative_folders(self, types="all"):
        """
        Create the custom folders for a BidsPatient.

        Parameters
        ----------
        types : str
            Which derivative folders to create. Can be "all", "preprocess", or "results".

        """
        if types not in ["all", "preprocess", "results"]:
            raise ValueError(
                "Can only create two types of derivative folders right now."
            )

        # create preprocess, results folders
        BidsBuilder.make_preprocessed_folder(
            self.bids_root, self.subject_id, self.session_id
        )
        BidsBuilder.make_results_folder(
            self.bids_root, self.subject_id, self.session_id
        )

    def get_subject_sessions(self):
        """
        Find all of the sessions that the BidsPatient has.

        Returns
        -------
        The list of sessions

        """
        return mne_bids.utils.get_entity_vals(self.subjdir, "ses")

    def get_subject_runs(self):
        """
        Find all the runs that the BidsPatient has.

        Returns
        -------
        list[str]
            The list of runs

        """
        return mne_bids.utils.get_entity_vals(self.subjdir, "run")

    def get_subject_kinds(self):
        """
        Find all the types of recordings that the BidsPatient has.

        Returns
        -------
        list[str]
            The list of recording types

        """
        return mne_bids.utils.get_kinds(self.subjdir)

    @property
    def numdatasets(self):
        """
        Find the number of datasets that the BidsPatient has.

        Returns
        -------
        int
            The number of datasets

        """
        return len(self.get_subject_runs())

    def _get_all_datasets(self):
        """Load in all the dataset paths."""
        # Should update after creating a run, but does not work currently
        edffiles = self.query.get(extension=".edf", return_type="file")
        fiffiles = self.query.get(extension=self.ext, return_type="file")
        self.edf_fpaths = edffiles
        self.dataset_fpaths = fiffiles

    def load_subject_metadata(self) -> Dict:
        """
        Find all the metadata corresponding to this subject.

        Returns
        -------
        dict
            A dictionary containing metadata information

        """
        participants_json = self.loader.load_participants_json()
        participants_tsv = self.loader.load_participants_tsv()

        # json file describes columns for this patient
        metadata = participants_tsv[
            participants_tsv["participant_id"]
            == make_bids_basename(subject=self.subject_id)
        ].to_dict("r")[0]
        # metadata = participants_tsv.set_index('participant_id')[self.subject_val].to_dict(orient="index")
        metadata["field_descriptions"] = participants_json
        return metadata

    def load_session(self, session_id, kind="eeg", extension="edf"):
        """
        Load in the all runs for the session.

        Parameters
        ----------
        session_id :
        kind :
        extension :

        Returns
        -------
        list[mne.io.BaseRaw]
            A list of objects with raw EEG data for a single session

        """
        # query for the run ids in the session based on factors
        runs_in_session = self.query.get(
            target="run",
            subject=self.subject_id,
            session=self.session_id,
            return_type="id",
            datatype=kind,
            extension=extension,
        )

        # load in all the mne_raw datasets
        rawlist = []
        for run_id in runs_in_session:
            loader = BidsLoader(bids_root=self.bids_root, bids_basename=None)
            rawlist.append(loader.load_dataset())
        return rawlist

    def load_run(self, session_id, run_id, kind="eeg"):
        """
        Load in the raw data for a specific run.

        Parameters
        ----------
        session_id :
        run_id :
        kind :

        Returns
        -------
        mne.IO.BaseRaw
            An object containing the raw EEG data

        """
        loader = BidsLoader(bids_root=self.bids_root, bids_basename=None)
        raw = loader.load_dataset()
        return raw

    def preprocess_edf_files(self, filelist: list = None, line_noise=60, kind="eeg"):
        """
        Convert .edf files to .fif and .json file pair.

        Parameters
        ----------
        filelist : list[os.PathLike]
            The list of all files to be converted into .fif format. Default is all edf files available
        line_noise : int
            The line noise to be eliminated. Typically 60 Hz in America

        """
        if filelist is None:
            filelist = self._find_non_processed_files()
        for i, fpath in enumerate(filelist):
            if not os.path.exists(fpath):
                raise OSError(f"{fpath} doesn't exist. Please pass in valid filepaths.")
            if line_noise != 50:
                loader = BidsLoader(bids_root=self.bids_root, bids_basename=None)
                sidecardict = loader.load_sidecar_json()
                sidecardict["PowerLineFrequency"] = line_noise

                writer = BidsWriter(bids_root=self.bids_root, bids_basename=None)
                writer.write_sidecar_json(sidecardict)

            # TODO: Check if .fif file already exists. Otherwise convert to fif

    def _find_non_processed_files(self):
        """
        Find the .edf files that have not been preprocessed into .fif files.

        Returns
        -------
        list:
            File paths for unprocessed recordings.

        """
        scans_df = self.loader.load_scans_tsv()
        colnames = scans_df.columns
        if "original_filename" not in colnames:
            return scans_df["filename"].tolist()
        return scans_df[scans_df["original_filename"] == ""]["filename"].tolist()

    def get_filesizes(self):
        """
        Get the total file size of all the files attached for this patient.

        Returns
        -------
        int
            size of all files

        """
        size = 0
        for f in self.dataset_fpaths:
            size += os.path.getsize(f)
        return size

    def _load_metadata(self, metadata: dict):
        """
        Load the passed metadata into memory.

        Parameters
        ----------
        metadata :

        """
        self.metadata = metadata

    def _create_metadata(self):
        """
        Create the metadata dictionary from whatever is in the participants.tsv file.

        Returns
        -------
        dict
            A dictionary representation of the metadata.

        """
        participants_df = self.loader.load_participants_tsv()
        participants_series = participants_df.loc[
            participants_df["participant_id"]
            == make_bids_basename(subject=self.subject_id)
        ].squeeze()
        series_dict = {"participant_id": participants_series.name}
        for index, value in participants_series.iteritems():
            series_dict[index] = value
        return series_dict

    def _write_to_participants_tsv(self, participants_df):
        participants_df.reset_index(inplace=True)
        self.writer.write_participants_tsv(participants_df)

    def add_participants_field(
        self,
        column_id,
        description: Union[str, dict],
        subject_val: str = "n/a",
        default_val: str = "n/a",
    ):
        """
        Add a field to the participants files (tsv and json).

        Parameters
        ----------
        column_id : str
            The new column to add to the participants.tsv file
        description: Union[str, dict]
            The new column's description to be added to
        subject_val : str
            The value of column_id for this BidsPatient
        default_val :
            The value of column_id to fill for the rest of subjects in participants.tsv

        """
        # write to participants json
        participants_dict = self.loader.load_participants_json()
        participants_df = self.loader.load_participants_tsv()
        participants_df = participants_df.set_index("participant_id")
        if column_id not in participants_dict.keys():
            participants_dict = BidsUtils.add_field_to_participants_json(
                participants_dict, column_id, description
            )
            self.writer.write_participants_json(participants_dict)

        # write to participants tsv file
        colnames = participants_df.columns
        if column_id not in colnames:
            add_list = [default_val] * participants_df.shape[0]
            participants_df[column_id] = add_list
            if subject_val != default_val:
                participants_df.at[
                    make_bids_basename(subject=self.subject_id), column_id
                ] = subject_val
            self._write_to_participants_tsv(participants_df)
        else:
            warnings.warn(
                "Field already exists for the participants file. Modifying instead"
            )
            self.modify_participants_file(column_id, subject_val)

        metadata = self._create_metadata()
        self._load_metadata(metadata)

    def remove_participants_field(self, column_id: str) -> Dict:
        """
        Remove a field from the participants files (tsv and json).

        Parameters
        ----------
        column_id : str
            The name of the column to remove from the json and tsv files

        """
        # modify at the JSON level
        participants_dict = self.loader.load_participants_json()
        participants_df = self.loader.load_participants_tsv()
        participants_df = participants_df.set_index("participant_id")
        if column_id not in participants_dict.keys():
            raise LookupError(
                f"The field {column_id} does not exist in the participants file"
            )
        participants_dict.pop(column_id, None)
        self.writer.write_participants_json(participants_dict)

        # modify at the TSV level
        colnames = participants_df.columns
        if column_id not in colnames:
            raise ValueError(
                f"The field {column_id} does not exist in the participants file"
            )
        participants_df.drop(column_id, axis=1, inplace=True)
        self._write_to_participants_tsv(participants_df)

        # recreate metadata
        metadata = self._create_metadata()
        self._load_metadata(metadata)

    def modify_participants_file(self, column_id: str, new_value: str):
        """
        Modify the BIDS dataset participants tsv file to include a new value.

        Parameters
        ----------
        column_id : str
            The name of the column to which data should be modified. Must exist in the tsv file
        new_value : str
            The new value of the column_id for this BidsPatient

        """
        # load in the data and do error check and set index
        participants_df = self.loader.load_participants_tsv()
        participants_df = participants_df.set_index("participant_id")
        colnames = participants_df.columns
        if column_id not in colnames:
            raise ValueError(
                f"{column_id} not in the existing participants file. Add it first."
            )

        # modify at the TSV level and reset_index
        participants_df.at[
            make_bids_basename(subject=self.subject_id), column_id
        ] = new_value
        self._write_to_participants_tsv(participants_df)

        # recreate metadata
        metadata = self._create_metadata()
        self._load_metadata(metadata)

    def _create_participants_json(self, fname, overwrite=False, verbose=False):
        """
        Create the participants json file for Bids compliance. an Extension of what is present in MNE-BIDS.

        TODO:
        1. determine how to incorporate into MNE-BIDS

        Parameters
        ----------
        fname : Union[str, os.PathLike]
            The path of the participants json file
        overwrite : bool
            Whether to overwrite an existing participants json file
        verbose : bool
            Whether to print the data from the participants json file to stdout

        """
        cols = OrderedDict()
        cols["participant_id"] = {"Description": "Unique participant identifier"}
        cols["age"] = {
            "Description": "Age of the participant at time of testing",
            "Units": "years",
        }
        cols["sex"] = {
            "Description": "Biological sex of the participant",
            "Levels": {"F": "female", "M": "male"},
        }
        _write_json(fname, cols, overwrite, verbose)
