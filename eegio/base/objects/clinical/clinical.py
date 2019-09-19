# -*- coding: utf-8 -*-
import os
from pprint import pprint

import numpy as np

from eegio.base.objects.clinical.baseclinical import DataSheet


def patient_check(f):
    def wrapper(*args):
        patientdict = args[0].patientdict
        patid = args[1]
        if patid not in patientdict.keys():
            raise RuntimeError(
                "{} is not in our clinical timeseries?! "
                "In Master_clinical.py".format(patid)
            )
        return f(*args)

    return wrapper


class DatasetClinical(DataSheet):
    """
    Dataset clinical metadata class for a single instance of a dataset.

    """

    def __init__(
        self,
        patid: str = None,
        datasetid=None,
        centerid=None,
        fpath: os.PathLike = None,
    ):
        super(DatasetClinical, self).__init__(fpath=fpath)
        self.patid = patid
        self.datasetid = datasetid
        self.centerid = centerid


class PatientClinical(DataSheet):
    def __init__(
        self,
        patid: str = None,
        datasetlist=[],
        centerid=None,
        fpath: os.PathLike = None,
    ):
        super(PatientClinical, self).__init__(fpath=fpath)
        self.patid = patid
        self.centerid = centerid
        self.datasetlist = datasetlist

        self._populate_attr_fromdf()

    def _populate_attr_fromdf(self):
        if "patient_id" in self.df.columns:
            self.patid = self.df.patient_id[0]

    def summary(self):
        summary_str = (
            f"{self.id} with {len(self.datasetlist)} datasets from center: {self.centerid}. "
            f""
        )
        pprint(summary_str)


# class MasterClinicalSheet(DataSheet):
#     """
#     Master clinical class. It wraps base patientclinical and datasetclinical objects
#     to create an easy to use object that grabs data from the cumulated patients and datasets
#
#     """
#
#     def __init__(self, patientlist=[], datasetlist=[]):
#         self.patients = patientlist
#         self.datasets = datasetlist
#
#         # create the objects
#         self._create_objs()
#         self._create_dictionary()
#
#     def summary(self):
#         pass
#
#     def _create_objs(self):
#         """
#         Helper function to extract metadata from the patient/timeseries clinical objects.
#
#         :return: Nothing
#         """
#         if all(isinstance(pat, PatientClinical) for pat in self.patients):
#             # get all unique patient and timeseries identifiers
#             self.patids = [pat.id for pat in self.patients]
#             self.centerids = np.unique([pat.centerid for pat in self.patients])
#         else:
#             self.patids = []
#             self.centerids = []
#
#         if all(isinstance(dataset, DatasetClinical) for dataset in self.datasets):
#             self.datasetids = [
#                 "-".join(dataset.id, dataset.datasetid) for dataset in self.datasets
#             ]
#         else:
#             self.datasetids = []
#
#     def _create_dictionary(self):
#         """
#         Converts passed list of datasets into dictionarys for faster access.
#
#         :return: Nothing
#         """
#         self.patientdict = {}
#         for pat in self.patients:
#             self.patientdict[pat.id] = pat
#
#         self.datasetdict = {}
#         for dataset in self.datasets:
#             self.datasetdict["-".join(dataset.id, dataset.datasetid)] = dataset
#
#     def get_center_patients(self, centerid):
#         # centernames = {
#         #     'nih': 'pt',
#         #     'ummc': 'ummc',
#         #     'jhu': 'jh',
#         #     'cleveland': 'la',
#         #     'clevelandnl': 'nl'
#         # }
#         allpats = [pat for pat in self.patients if pat.centerid == centerid]
#         return allpats
#
#     @patient_check
#     def get_patient_clinicaldiff(self, patid):
#         return self.patientdict[patid].clinical_difficulty
#
#     @patient_check
#     def get_patient_cezlobe(self, patid):
#         return self.patientdict[patid].cezlobe
#
#     @patient_check
#     def get_patient_age(self, patid):
#         return self.patientdict[patid].age
#
#     @patient_check
#     def get_patient_center(self, patid):
#         # get outcome, engel score
#         return self.patientdict[patid].clinical_center
#
#     @patient_check
#     def get_patient_onsetage(self, patid):
#         # get outcome, engel score
#         return self.patientdict[patid].onsetage
#
#     @patient_check
#     def get_patient_handedness(self, patid):
#         # get outcome, engel score
#         return self.patientdict[patid].handedness
#
#     @patient_check
#     def get_patient_gender(self, patid):
#         # get outcome, engel score
#         return self.patientdict[patid].gender
#
#     @patient_check
#     def get_patient_outcome(self, patid):
#         # get outcome, engel score
#         return self.patientdict[patid].outcome
#
#     @patient_check
#     def get_patient_modality(self, patid):
#         # get outcome, engel score
#         return self.patientdict[patid].modality
#
#     @patient_check
#     def get_patient_engelscore(self, patid):
#         # get outcome, engel score
#         return self.patientdict[patid].engel_score
#
#     @patient_check
#     def get_patient_ilaescore(self, patid):
#         # get outcome, engel score
#         return self.patientdict[patid].ilae_score
#
#     @patient_check
#     def get_patient_dataset_semiology(self, patid, datasetid, type="sz"):
#         if type == "sz":
#             return self.datasets[patid + datasetid.replace("_", "")].seizure_semiology
#         elif type == "clinical":
#             return self.datasets[patid + datasetid.replace("_", "")].clinical_semiology
#
#     @patient_check
#     def get_patient_dataset_ezhypo(self, patid, datasetid):
#         return self.datasets[patid + datasetid.replace("_", "")].ez_hypo_contacts
#
#     @patient_check
#     def get_patient_resectedcontacts(self, patid):
#         return self.patientdict[patid].resected_contacts
#
#     @patient_check
#     def get_patient_ablatedcontacts(self, patid):
#         return self.patientdict[patid].ablated_contacts
#
#     def get_all_success(self):
#         patids = []
#         for patid in self.patientdict.keys():
#             outcome = self.patientdict[patid].outcome
#             if outcome == "s":
#                 patids.append(patid)
#         return patids
#
#     def get_all_failure(self):
#         patids = []
#         for patid in self.patientdict.keys():
#             outcome = self.patientdict[patid].outcome
#             if outcome == "f":
#                 patids.append(patid)
#         return patids
#
#     def get_all_nonresection(self):
#         patids = []
#         for patid in self.patientdict.keys():
#             outcome = self.patientdict[patid].outcome
#             if outcome not in ["s", "f"]:
#                 patids.append(patid)
#         return patids


if __name__ == "__main__":
    patid = "pat01"
    datasetid = "sz_2"

    datasetids = ["sz_2", "sz_3"]
    centerid = "jhu"
    example_datadict = {
        "length_of_recording": 400,
        "timepoints": np.hstack((np.arange(0, 100), np.arange(5, 105))),
    }

    patclin = PatientClinical(patid, datasetids, centerid)
    patclin.load_from_dict(example_datadict)

    clin = MasterClinicalSheet([patclin])
    clin.get_patient_clinicaldiff("hi")
    print(clin.get_patient_clinicaldiff(patid))

    filepath = "../../../tests/organized_datasheet_formatted.xlsx"

    patid = "pat01"
    datasetid = "sz_2"

    datasetids = ["sz_2", "sz_3"]
    centerid = "jhu"
    example_datadict = {
        "length_of_recording": 400,
        "timepoints": np.hstack((np.arange(0, 100), np.arange(5, 105))),
    }

    patclin = PatientClinical(patid, datasetids, centerid)
    patclin.load_from_dict(example_datadict)

    for key, val in example_datadict.items():
        testval = getattr(patclin, key)
        # assert testval == val

    print(dir(patclin))
