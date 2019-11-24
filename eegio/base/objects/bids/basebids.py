from abc import abstractmethod, abstractclassmethod, abstractproperty, abstractstaticmethod

import os
from typing import Union
import warnings
import pandas as pd
import json
import bids
import bids.reports
import mne_bids
from mne_bids.utils import print_dir_tree


class BaseBids(object):
    """
    An instantiator class of the bids format that any BIDS derivative would use. It handles:
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
        The modality of data we are reading. For example, EEG, MRI, CT, etc.

    Notes
    -----
    datadir should be named and formatted according to BIDS format. For example:

    datadir = <study_name>/raw/

    maybe it will be:

    datadir = <study_name>/derivatives/<derivative_category_name>

    Examples
    --------
    """

    def __init__(
            self,
            datadir: Union[os.PathLike, str],
            modality: str,
    ):
        self.rootdir = datadir
        self.datadir = datadir
        self.modality = modality

        # initialize a dictionary to store all of our paramters
        self.parameter_dict = {}

        # store parameters
        self._store_params()

        if not os.path.exists(self.description_fpath):
            self.create_dataset_folder(self.datadir,
                                       name="BIDS_DATASET")

        if not os.path.exists(self.participants_tsv_fpath):
            self.create_participants_files(self.participants_tsv_fpath, self.participants_json_fpath)

    @property
    def description_fpath(self):
        return os.path.join(self.datadir, "dataset_description.json")

    @property
    def participants_tsv_fpath(self):
        return os.path.join(self.datadir, "participants.tsv")

    @property
    def participants_json_fpath(self):
        return os.path.join(self.datadir, "participants.json")

    @property
    def query(self):
        return bids.BIDSLayout(self.datadir)

    @property
    def parameters(self):
        return self.parameter_dict

    @property
    def data_types(self):
        return mne_bids.utils.get_kinds(self.datadir)

    @property
    def subject_nums(self):
        return mne_bids.utils.get_entity_vals(self.datadir, 'sub')

    @property
    def num_subjects(self):
        return len(self.subject_nums)

    def print_dir_tree(self):
        basepath = os.path.join(self.datadir)
        print_dir_tree(basepath)

    def _store_params(self):
        self.parameter_dict.update(
            datadir=self.datadir,
            modality=self.modality,
        )

    def create_dataset_folder(self, fpath, **dataset_description_kwargs):
        # create dataset description
        mne_bids.make_dataset_description(fpath, **dataset_description_kwargs)

    def create_participants_files(self, tsv_fpath, json_fpath):
        empty_df = pd.DataFrame(columns=['participant_id', 'age', 'sex'])
        empty_df.to_csv(tsv_fpath, sep='\t', index=False)
        participants_json = {'participant_id': 'Id of the subject', 'age': 'age of subject in years',
                             'sex': 'Male (M) or female (F)'}
        with open(json_fpath, 'w') as fp:
            json.dump(participants_json, fp, indent=4)
            fp.close()


    def print_summary(self):
        """
        Print a summary of the patient in the form of a directory tree showing
        all available data
        """
        if self.modality is not "MRI":
            warnings.warn("Print summary only works for neuroimaging type datasets.")
            return
        report = bids.reports.BIDSReport(self.query)
        counter = report.generate()
        main_report = counter.most_common()
        print(main_report)

    # def