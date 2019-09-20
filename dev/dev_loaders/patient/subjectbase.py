import itertools
import os
import warnings

import numpy as np
from eegio.base import config
from natsort import natsorted

from eegio.loaders.base.baseio import BaseIO


class Subject:
    def __init__(
        self, root_dir, subjid, datatype, userid: str = "", clinical_center: str = None
    ):
        super(Subject, self).__init__(root_dir=root_dir)

        self.subjid = subjid
        self.datatype = datatype
        self.userid = userid
        self.clinical_center = clinical_center

        self.is_loaded = False
        self.datasets = []

        # return data structures of the results
        self.datasets = []
        self.dataset_ids = []
        self.seizure_datasets = []
        self.interictal_datasets = []
        self.stimulation_datasets = []

    def _getalljsonfilepaths(self, eegdir, subjid=None):
        pass

    def _initialize_patient_metadata(self):
        """
        Loop through available metadata in .json files and accumulate various metadata points about the datasets
        available for this patient.

        :return:
        """
        pass

    def summary(self):
        summary_str = (
            f"{self.subjid} managed by: {self.userid} at {self.clinical_center}."
        )
        print(summary_str)
        return summary_str

    def get_datasets(self):
        return self.datasets

    def get_all_dataset_filepaths(self):
        pass
        fpaths = []
        natsorted(fpaths)

    def load_dataset(self, filepath: os.PathLike):
        pass

    def load_all_datasets(self, reload=False, cache_results=True):
        """
        Function to load all the datasets in their objects and return a list of them.

        :return:
        """
        all_dataset_filepaths = self.get_all_dataset_filepaths()

        if len(all_dataset_filepaths) > 3:
            mb = "NOT SET YET"
            warnings.warn(
                f"You are about to load more then 3 datasets with {mb} of projected"
                f" data."
            )
        if self.is_loaded:
            raise RuntimeError(
                "Patient already loaded all the datasets. Access them through"
                " the datasets attribute, or get_datasets() function. If want to reload, pass in reload=True."
            )
        datasets = []

        if not reload:
            for fpath in all_dataset_filepaths:
                datasetobj = self.load_dataset(fpath)
                datasets.append(datasetobj)

        if cache_results:
            self.is_loaded = True
            self.datasets = datasets

        return datasets
