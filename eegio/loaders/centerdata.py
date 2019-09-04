import itertools
import os

import numpy as np

from base.config.dataconfig import MODELDATATYPES, RAWMODALITIES
from base.utils.data_structures_utils import ensure_list
from eegio.loaders.base.basemultipleloader import BaseMultipleDatasetLoader
from eegio.loaders.patient.subjectrawloader import SubjectRawLoader
from eegio.loaders.patient.subjectresultsloader import SubjectResultsLoader


class CenterLoader(BaseMultipleDatasetLoader):
    def __init__(self, root_dir, centername,
                 datatype='frag', subjids=[],
                 preload=True):
        super(CenterLoader, self).__init__(root_dir=root_dir)
        self.centername = centername
        self.subjids = ensure_list(subjids)
        self.datatype = datatype

        self.loadingfunc = None
        if self.datatype in MODELDATATYPES:
            self.loadingfunc = SubjectResultsLoader
        elif self.datatype in RAWMODALITIES:
            self.loadingfunc = SubjectRawLoader

        if subjids == []:
            self._getall_subjsonfilepaths()
            self._getallsubjectnames()

        if preload:
            self.read_all_files()

    def _getall_subjsonfilepaths(self):
        self.jsonfilepaths = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.json') \
                        and not file.startswith('.'):
                    self.jsonfilepaths.append(os.path.join(root, file))

    def _getallsubjectnames(self):
        for jsonfilepath in self.jsonfilepaths:
            jsonfilename = os.path.basename(jsonfilepath)

            patid = jsonfilename.split('_')[0]
            self.subjids.append(patid)
        self.subjids = np.unique(self.subjids)

    def read_all_files(self):
        centerdatasets = []
        centerdatasetids = []
        centerpatientids = []
        outcome_list = []
        engelscore_list = []
        clindifficulty_list = []

        # loop through subject ids
        for subjid in self.subjids:
            # print("Loading in: ", subjid)
            subjloader = self.loadingfunc(root_dir=self.root_dir,
                                          subjid=subjid,
                                          datatype=self.datatype,
                                          preload=False)
            subjloader.read_all_files()

            # extract results for subject
            subjectdatasets = subjloader.datasets
            dataset_ids = subjloader.dataset_ids

            centerdatasets.extend(subjectdatasets)
            centerdatasetids.extend(dataset_ids)
            centerpatientids.extend(
                list(itertools.repeat(subjid, len(dataset_ids))))

            outcome = subjloader.get_outcomes(returnlist=True)
            engel_score = subjloader.get_engelscores(returnlist=True)
            clindiff = subjloader.get_clinicaldifficulty(returnlist=True)
            clindifficulty_list.extend(clindiff)
            outcome_list.extend(outcome)
            engelscore_list.extend(engel_score)

        print(engelscore_list)
        print(clindifficulty_list)

        self.datasets = np.array(centerdatasets)
        self.datasetids = np.array(centerdatasetids)
        self.patientids = np.array(centerpatientids)
        self.engelscore_list = np.array(engelscore_list)
        self.clindifficulty_list = np.array(clindifficulty_list)
        self.outcome_list = np.array(outcome_list)

        return centerdatasets, centerpatientids, centerdatasetids

    def get_patient_data(self):
        pass
