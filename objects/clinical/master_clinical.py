# -*- coding: utf-8 -*-
import numpy as np
import warnings
import ast

from eztrack.eegio.objects.clinical.baseclinical import PatientClinical, DatasetClinical

from ast import literal_eval

def format_list_str_channels(chanlist):
    return literal_eval(chanlist[0])

class MasterClinicalSheet():
    """
    Master clinical class. It wraps base patientclinical and datasetclinical objects
    to create an easy to use object that grabs data per patient and/or dataset.

    """

    def __init__(self, ieeg_meta, dataset_meta, scalp_meta):
        # store a list copy of all the dataframes
        self.patient_df = ieeg_meta
        self.dataset_df = dataset_meta
        self.scalp_df = scalp_meta
        self.dfs = [ieeg_meta, dataset_meta, scalp_meta]

        self.patients = dict()
        self.datasets = dict()

        # create the objects
        self._create_patientobjs()
        self._create_datasetobjs()
        self._merge_scalp_information()

    def _ensure_list_of_strs(self):
        joinstr = lambda x: ",".join(x)

        ensure_strs = [
            "resected_contacts",
            "ablated_contacts",
            "wm_contacts",
            "bad_channels"
        ]
        for col in ensure_strs:
            self.patient_df[col] = self.patient_df[col].astype(str).apply(ast.literal_eval).apply(joinstr)

    def _create_patientobjs(self):
        # get all unique patient identifiers
        unique_patids = self.patient_df['patient_id'].values

        # all rows in this df should be unique...
        if len(np.unique(unique_patids)) != len(unique_patids):
            raise RuntimeError("IEEG Dataframe Excel sheet should have one unique patient identifier"
                               "per row. Check Excel sheet! ")
        unique_patids = np.unique(unique_patids)

        self._ensure_list_of_strs()

        # create patient objects
        for patid in unique_patids:
            # get pat datraframe for this specific patient
            patdf = self.patient_df[self.patient_df['patient_id'] == patid]
            patobj = PatientClinical(patid=patid, df=patdf)

            # store that patient
            self.patients[patid] = patobj

    def _create_datasetobjs(self):
        # get the patient and dataset identifiers
        patids = self.dataset_df['patient_id'].values
        datasetids = self.dataset_df['dataset_id'].values
        fullids = zip(patids, datasetids)

        # create dataset objects
        for patid, datasetid in fullids:
            # get pat datraframe for this specific patient
            datasetdf = self.dataset_df[(self.dataset_df['patient_id'] == patid) &
                                        (self.dataset_df['dataset_id'] == datasetid)]

            if datasetdf.empty:
                raise RuntimeError("{} with {} has an empty dataframe row returned. "
                                   "Make sure to reorganize the corresponding data point in Excel.".format(
                                       patid, datasetid
                                   ))

            # create the datasetobject
            datasetobj = DatasetClinical(
                patid=patid, datasetid=datasetid, df=datasetdf)

            # store that patient
            self.datasets[patid + datasetid.replace('_', '')] = datasetobj

    def _merge_scalp_information(self):
        # create patient objects
        for i, pat in enumerate(self.patients.keys()):
            patid = self.patients[pat].patient_id

            # get pat datraframe for this specific patient
            scalpdf = self.scalp_df[self.scalp_df['patient_id'] == patid]

            if scalpdf.empty:
                # print(patid, " is empty.")
                warnings.warn("No scalp information for this patient {}".format(
                    patid))
                continue

            # modify the patient dataframe
            # print(scalpdf['cezlobe'].values)
            # print(scalpdf['cezlobe'].values[0])
            self.patients[pat].df['cezlobe'] = scalpdf['cezlobe'].values
            # self.patients[pat].df['cezlobe'] = format_list_str_channels(scalpdf['cezlobe'].values)
            # print(scalpdf['cezlobe'].values)

    def get_center_patients(self, centerid):
        centernames = {
            'nih': 'pt',
            'ummc': 'ummc',
            'jhu': 'jh',
            'cleveland': 'la',
            'clevelandnl': 'nl'
        }
        nameprefix = centernames[centerid]
        # get succes_pat and failed_pat
        success_pats = self.patient_df.loc[(self.patient_df['outcome'] == 's') &
                                           (self.patient_df['patient_id'].str.contains(nameprefix))]['patient_id'].values
        fail_pats = self.patient_df.loc[(self.patient_df['outcome'] == 'f') &
                                        (self.patient_df['patient_id'].str.contains(nameprefix))]['patient_id'].values
        nr_pats = self.patient_df.loc[(self.patient_df['outcome'] == 'nr') &
                                      (self.patient_df['patient_id'].str.contains(nameprefix))]['patient_id'].values
        allpats = np.concatenate((success_pats, fail_pats))

        allpats = np.delete(allpats, np.where((allpats == 'la01_2')))
        #     allpats = np.delete(allpats, np.where((allpats == 'la06')))
        #     allpats = np.delete(allpats,
        #                         np.where((allpats == la for la in ['la02' 'la03' 'la05' 'la07' 'la21' 'la22' 'la23' 'la01'])))
        return allpats

    def get_patient_cezlobe(self, patid):
        if patid not in self.patients.keys():
            raise RuntimeError("{} is not in our clinical dataset?! "
                               "In Master_clinical.py".format(patid))

        return self.patients[patid].cezlobe

    def get_patient_clinicaldiff(self, patid):
        if patid not in self.patients.keys():
            raise RuntimeError("{} is not in our clinical dataset?! "
                               "In Master_clinical.py".format(patid))

        return self.patients[patid].clinical_difficulty

    def get_patient_age(self, patid):
        # get outcome, engel score
        if patid not in self.patients.keys():
            raise RuntimeError("{} is not in our clinical dataset?! "
                               "In Master_clinical.py".format(patid))
        return self.patients[patid].age

    def get_patient_center(self, patid):
        # get outcome, engel score
        if patid not in self.patients.keys():
            raise RuntimeError("{} is not in our clinical dataset?! "
                               "In Master_clinical.py".format(patid))
        return self.patients[patid].clinical_center

    def get_patient_onsetage(self, patid):
        # get outcome, engel score
        if patid not in self.patients.keys():
            raise RuntimeError("{} is not in our clinical dataset?! "
                               "In Master_clinical.py".format(patid))
        return self.patients[patid].onsetage

    def get_patient_handedness(self, patid):
        # get outcome, engel score
        if patid not in self.patients.keys():
            raise RuntimeError("{} is not in our clinical dataset?! "
                               "In Master_clinical.py".format(patid))
        return self.patients[patid].handedness

    def get_patient_gender(self, patid):
        # get outcome, engel score
        if patid not in self.patients.keys():
            raise RuntimeError("{} is not in our clinical dataset?! "
                               "In Master_clinical.py".format(patid))
        return self.patients[patid].gender

    def get_patient_outcome(self, patid):
        # get outcome, engel score
        if patid not in self.patients.keys():
            raise RuntimeError("{} is not in our clinical dataset?! "
                               "In Master_clinical.py".format(patid))

        return self.patients[patid].outcome

    def get_patient_modality(self, patid):
        # get outcome, engel score
        if patid not in self.patients.keys():
            raise RuntimeError("{} is not in our clinical dataset?! "
                               "In Master_clinical.py".format(patid))

        return self.patients[patid].modality


    def get_patient_engelscore(self, patid):
        # get outcome, engel score
        if patid not in self.patients.keys():
            raise RuntimeError("{} is not in our clinical dataset?! "
                               "In Master_clinical.py".format(patid))

        return self.patients[patid].engel_score

    def get_patient_ilaescore(self, patid):
        # get outcome, engel score
        if patid not in self.patients.keys():
            raise RuntimeError("{} is not in our clinical dataset?! "
                               "In Master_clinical.py".format(patid))

        return self.patients[patid].ilae_score

    def get_patient_dataset_semiology(self, patid, datasetid, type='sz'):
        if type == 'sz':
            return self.datasets[patid + datasetid.replace('_', '')].seizure_semiology
        elif type == 'clinical':
            return self.datasets[patid + datasetid.replace('_', '')].clinical_semiology

    def get_patient_dataset_ezhypo(self, patid, datasetid):
        return self.datasets[patid + datasetid.replace('_', '')].ez_hypo_contacts

    def get_patient_resectedcontacts(self, patid):
        return self.patients[patid].resected_contacts

    def get_patient_ablatedcontacts(self, patid):
        return self.patients[patid].ablated_contacts

    def get_all_success(self):
        patids = []
        for patid in self.patients.keys():
            outcome = self.patients[patid].outcome
            if outcome == 's':
                patids.append(patid)
        return patids

    def get_all_failure(self):
        patids = []
        for patid in self.patients.keys():
            outcome = self.patients[patid].outcome
            if outcome == 'f':
                patids.append(patid)
        return patids

    def get_all_nonresection(self):
        patids = []
        for patid in self.patients.keys():
            outcome = self.patients[patid].outcome
            if outcome not in ['s', 'f']:
                patids.append(patid)
        return patids

    def split_by_engelscore(self):
        pass

    def split_by_clinicaldiff(self):
        pass
