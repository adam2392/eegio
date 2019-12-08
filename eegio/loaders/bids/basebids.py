"""
Authors: Adam Li and Patrick Myers.

Version: 1.0
"""

import os
import warnings
from collections import OrderedDict
from typing import Union

import bids
import bids.reports
import mne_bids
from mne_bids.utils import print_dir_tree

from eegio.base.utils.bids_helper import write_json


class BaseBids(object):
    """
    An instantiator class of the bids format that any BIDS derivative would use.

    It handles:
     - the logistics of creating a bids format file structure
     - extracting metadata from files stored in iEEG-BIDS format
     - easy loading of data files (per dataset, per patient, per derivative (i.e. result), etc.)

    Only supports iEEG/scalp EEG data. We assume you store data as a BIDS with sub_<###> as the
    identifiers. Allows easy querying of its own data structure via:

    self.query.

    TODO:
    1. Add support for runs and acquisitions (https://mne.tools/mne-bids/generated/mne_bids.utils.get_entity_vals.html#mne_bids.utils.get_entity_vals)
    2.

    Attributes
    ----------
    datadir: os.PathLike
        The base directory for storing data
    modality: str
        The kind of data we are reading. For example, EEG, MRI, CT, etc.

    Notes
    -----
    datadir should be named and formatted according to BIDS format. For example:

    datadir = <study_name>/raw/

    maybe it will be:

    datadir = <study_name>/derivatives/<derivative_category_name>

    """

    def __init__(self, datadir: Union[os.PathLike, str], modality: str):
        """
        Initialize a BaseBids Object.

        Parameters
        ----------
        datadir :
        modality :

        """
        self.rootdir = datadir
        self.datadir = datadir
        self.modality = modality

        # initialize a dictionary to store all of our paramters
        self.parameter_dict = {}

        # store parameters
        self._store_params()

        if not os.path.exists(self.description_fpath):
            self.create_dataset_folder(self.datadir, name="BIDS_DATASET")

        # if not os.path.exists(self.participants_tsv_fpath):
        #     self.create_participants_files(self.participants_tsv_fpath, self.participants_json_fpath)

    @property
    def description_fpath(self):
        """
        Get the path of the dataset description file.

        Returns
        -------
        Dataset description file path.

        """
        return os.path.join(self.datadir, "dataset_description.json")

    @property
    def participants_tsv_fpath(self):
        """
        Get the path of the participants tsv file.

        Returns
        -------
        Participants tsv file path.

        """
        return os.path.join(self.datadir, "participants.tsv")

    @property
    def participants_json_fpath(self):
        """
        Get the path of the participants json file.

        Returns
        -------
        Participants json file path.

        """
        return os.path.join(self.datadir, "participants.json")

    @property
    def query(self):
        """
        Get a queryable object representing the entire Bids dataset.

        Returns
        -------
        A BidsLayout object

        """
        return bids.BIDSLayout(self.datadir)

    @property
    def parameters(self):
        """
        Get a dictionary of parameters for this Bids object.

        Returns
        -------
        The parameter dict.

        """
        return self.parameter_dict

    @property
    def data_types(self):
        """
        Get a list of all recording types in this Bids dataset.

        Returns
        -------
        A list of recording types

        """
        return mne_bids.utils.get_kinds(self.datadir)

    @property
    def subject_nums(self):
        """
        Get a list of subject_ids in this Bids dataset.

        Returns
        -------
        A list of subject_val values.

        """
        return mne_bids.utils.get_entity_vals(self.datadir, "sub")

    @property
    def num_subjects(self):
        """
        Get the number of subjects in this Bids dataset.

        Returns
        -------
        The number of subjects

        """
        return len(self.subject_nums)

    def print_dir_tree(self):
        """Print a directory tree for this Bids dataset."""
        basepath = os.path.join(self.datadir)
        print_dir_tree(basepath)

    def _store_params(self):
        """Store parameters into a dict object."""
        self.parameter_dict.update(datadir=self.datadir, modality=self.modality)

    def create_dataset_folder(self, fpath, **dataset_description_kwargs):
        """
        Create a new Bids format dataset.

        Parameters
        ----------
        fpath : Union[str, os.PathLike]
            The path of this new directory
        dataset_description_kwargs : dict
            Other arguments used by make_dataset_description

        """
        # create dataset description
        mne_bids.make_dataset_description(fpath, **dataset_description_kwargs)

    def print_summary(self):
        """Print a summary of the patient in the form of a directory tree showing all available data."""
        if self.modality != "MRI":
            warnings.warn("Print summary only works for neuroimaging type datasets.")
            return
        report = bids.reports.BIDSReport(self.query)
        counter = report.generate()
        main_report = counter.most_common()
        print(main_report)

    def _create_participants_json(self, fname, overwrite=False, verbose=False):
        """
        Create the participants json file for Bids compliance.

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

        write_json(fname, cols, overwrite, verbose)

    # def _participants_tsv(self, raw, subject_val, fname, overwrite=False,
    #                       verbose=True):
    #     """Create a participants.tsv file and save it.
    #
    #     This will append any new participant data to the current list if it
    #     exists. Otherwise a new file will be created with the provided information.
    #
    #     Parameters
    #     ----------
    #     raw : instance of Raw
    #         The data as MNE-Python Raw object.
    #     subject_val : str
    #         The subject name in BIDS compatible format ('01', '02', etc.)
    #     fname : str
    #         Filename to save the participants.tsv to.
    #     overwrite : bool
    #         Whether to overwrite the existing file.
    #         Defaults to False.
    #         If there is already data for the given `subject_val` and overwrite is
    #         False, an error will be raised.
    #     verbose : bool
    #         Set verbose output to true or false.
    #
    #     """
    #     subject_val = 'sub-' + subject_val
    #     data = OrderedDict(participant_id=[subject_val])
    #
    #     subject_age = "n/a"
    #     sex = "n/a"
    #     subject_info = raw.info['subject_info']
    #     if subject_info is not None:
    #         sexes = {0: 'n/a', 1: 'M', 2: 'F'}
    #         sex = sexes[subject_info.get('sex', 0)]
    #
    #         # determine the age of the participant
    #         age = subject_info.get('birthday', None)
    #         meas_date = raw.info.get('meas_date', None)
    #         if isinstance(meas_date, (tuple, list, np.ndarray)):
    #             meas_date = meas_date[0]
    #
    #         if meas_date is not None and age is not None:
    #             bday = datetime(age[0], age[1], age[2])
    #             meas_datetime = datetime.fromtimestamp(meas_date)
    #             subject_age = _age_on_date(bday, meas_datetime)
    #         else:
    #             subject_age = "n/a"
    #
    #     data.update({'age': [subject_age], 'sex': [sex]})
    #
    #     if os.path.exists(fname):
    #         orig_data = _from_tsv(fname)
    #         # whether the new data exists identically in the previous data
    #         exact_included = _contains_row(orig_data,
    #                                        {'participant_id': subject_val,
    #                                         'age': subject_age,
    #                                         'sex': sex})
    #         # whether the subject id is in the previous data
    #         sid_included = subject_val in orig_data['participant_id']
    #         # if the subject data provided is different to the currently existing
    #         # data and overwrite is not True raise an error
    #         if (sid_included and not exact_included) and not overwrite:
    #             raise FileExistsError('"%s" already exists in the participant '  # noqa: E501 F821
    #                                   'list. Please set overwrite to '
    #                                   'True.' % subject_val)
    #         # otherwise add the new data
    #         data = _combine(orig_data, data, 'participant_id')
    #
    #     # overwrite is forced to True as all issues with overwrite == False have
    #     # been handled by this point
    #     _write_tsv(fname, data, True, verbose)
    #
    #     return fname
    #
    # def _participants_json(self, fname, overwrite=False, verbose=True):
    #     """Create participants.json for non-default columns in accompanying TSV.
    #
    #     Parameters
    #     ----------
    #     fname : str
    #         Filename to save the scans.tsv to.
    #     overwrite : bool
    #         Defaults to False.
    #         Whether to overwrite the existing data in the file.
    #         If there is already data for the given `fname` and overwrite is False,
    #         an error will be raised.
    #     verbose : bool
    #         Set verbose output to true or false.
    #
    #     """
    #     cols = OrderedDict()
    #     cols['participant_id'] = {'Description': 'Unique participant identifier'}
    #     cols['age'] = {'Description': 'Age of the participant at time of testing',
    #                    'Units': 'years'}
    #     cols['sex'] = {'Description': 'Biological sex of the participant',
    #                    'Levels': {'F': 'female', 'M': 'male'}}
    #
    #     write_json(fname, cols, overwrite, verbose)
    #
    #     return fname
