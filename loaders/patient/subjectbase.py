import os

from natsort import natsorted

from eztrack.base.config.dataconfig import ICTAL_TYPES, INTERICTAL_TYPES
from eztrack.eegio.loaders.base.basemultipleloader import BaseMultipleDatasetLoader


class SubjectBase(BaseMultipleDatasetLoader):
    def __init__(self, root_dir, subjid, datatype):
        super(SubjectBase, self).__init__(root_dir=root_dir)

        self.subjid = subjid
        self.datatype = datatype

        self.is_loaded = False

        # return data structures of the results
        self.datasets = []
        self.dataset_ids = []

        self.seizure_datasets = []
        self.interictal_datasets = []
        self.stimulation_datasets = []

    def _getalljsonfilepaths(self, eegdir, subjid=None):
        """
        Helper function to get all the jsonfile names for
        user to choose one
        """
        jsonfilepaths = [f for f in os.listdir(eegdir)
                         if f.endswith('.json')
                         if not f.startswith('.')]

        if subjid is not None:
            jsonfilepaths = [f for f in jsonfilepaths if subjid + '_' in f]
        # jsonfilepaths = [f for f in jsonfilepaths if self.datatype in f]

        self.jsonfilepaths = natsorted(jsonfilepaths)

    def __len__(self):
        return len(self.jsonfilepaths)

    def __str__(self):
        return "{} - {} datasets".format(self.patient, len(self.jsonfilepaths))

    def __repr__(self):
        return "{} - {} datasets".format(self.patient, len(self.jsonfilepaths))

    @property
    def size(self):
        return len(self.jsonfilepaths)

    @property
    def patient(self):
        return self.subjid

    def get_seizuredata(self):
        return self.seizure_datasets

    def get_interictaldata(self):
        return self.interictal_datasets

    def get_stimulationdata(self):
        return self.stimulation_datasets

    def split_files_type(self):
        """
        Method to split files by their dataset type (e.g. by seizure and non-seizures).

        :return:
        """
        for idx, datasetid in enumerate(self.dataset_ids):
            if datasetid not in ICTAL_TYPES and datasetid not in INTERICTAL_TYPES:
                raise ValueError("dataset ids were not named properly! They should"
                                 "be part of ictal or interictal types. They are: {} and {}".format(ICTAL_TYPES,
                                                                                                    INTERICTAL_TYPES))

            if datasetid == 'ii' or 'interictal' in datasetid:
                self.interictal_datasets.append(self.datasets[idx])

            elif datasetid == 'stimulated seizure':
                self.stimulation_datasets.append(self.datasets[idx])

            elif datasetid == 'sz' or 'seizure' in datasetid:
                self.seizure_datasets.append(self.datasets[idx])
