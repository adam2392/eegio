"""
Authors: Adam Li and Patrick Myers.

Version: 1.0
"""

import os
import warnings

import bids
import bids.reports
import mne_bids
from mne_bids.utils import _write_json
from mne_bids.utils import print_dir_tree


class BaseBids(object):
    """
    An instantiator class of the bids format that extends usage of BIDS.

    Any BIDS derivative would use. It handles:
     - the logistics of creating a bids format file structure
     - extracting metadata from files stored in iEEG-BIDS format
     - easy loading of data files (per dataset, per patient, per derivative (i.e. result), etc.)

    Only supports iEEG/scalp EEG data. We assume you store data as a BIDS with sub_<###> as the
    identifiers. Allows easy querying of its own data structure via:
    - self.query

    Attributes
    ----------
    bids_root: os.PathLike
        The path of the root of the BIDS compatible folder. The session and
        subject specific folders will be populated automatically by parsing
        bids_basename.

    Notes
    -----
    tmp_bids_root should be named and formatted according to BIDS format. For example:

    tmp_bids_root = <study_name>/raw/

    maybe it will be:

    tmp_bids_root = <study_name>/derivatives/<derivative_category_name>

    """

    def __init__(self, bids_root):
        self.bids_root = bids_root

        if not os.path.exists(self.description_fpath):
            self.create_dataset_description(self.bids_root, name="BIDS_DATASET")

    @property
    def description_fpath(self):
        """
        Get the path of the dataset description file.

        Returns
        -------
        Dataset description file path.

        """
        return os.path.join(self.bids_root, "dataset_description.json")

    @property
    def participants_tsv_fpath(self):
        """
        Get the path of the participants tsv file.

        Returns
        -------
        Participants tsv file path.

        """
        return os.path.join(self.bids_root, "participants.tsv")

    @property
    def participants_json_fpath(self):
        """
        Get the path of the participants json file.

        Returns
        -------
        Participants json file path.

        """
        return os.path.join(self.bids_root, "participants.json")

    @property
    def query(self):
        """
        Get a queryable object representing the entire Bids dataset.

        Returns
        -------
        A BidsLayout object

        """
        return bids.BIDSLayout(self.bids_root)

    @property
    def data_types(self):
        """
        Get a list of all recording types in this Bids dataset.

        Returns
        -------
        A list of recording types

        """
        return mne_bids.utils.get_kinds(self.bids_root)

    @property
    def subject_nums(self):
        """
        Get a list of subject_ids in this Bids dataset.

        Returns
        -------
        A list of subject_val values.

        """
        return mne_bids.utils.get_entity_vals(self.bids_root, "sub")

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
        basepath = os.path.join(self.bids_root)
        print_dir_tree(basepath)

    def create_dataset_description(self, fpath, **dataset_description_kwargs):
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
        """Print a summary of the patient in the form of a directory tree."""
        if "MRI" not in self.query.get_entities(self.bids_root):
            warnings.warn("Print summary only works for neuroimaging type datasets.")
            return
        report = bids.reports.BIDSReport(self.query)
        counter = report.generate()
        main_report = counter.most_common()
        print(main_report)
