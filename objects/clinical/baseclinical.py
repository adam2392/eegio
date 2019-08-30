# -*- coding: utf-8 -*-
from ast import literal_eval
import ast

def format_list_str_channels(chanlist):
    return literal_eval(chanlist[0])

class PatientClinical():
    """
    Patient clinical metadata class for a single instance of a patient.

    """

    def __init__(self, patid, df):
        self.id = patid
        self.df = df

        # self._ensure_list_of_strs()

    @property
    def modality(self):
        return self.df['modality'].values[0]

    @property
    def patient_id(self):
        return self.df['patient_id'].values[0]

    @property
    def outcome(self):
        return self.df['outcome'].values[0]

    @property
    def engel_score(self):
        return self.df['engel_score'].values[0]

    @property
    def ilae_score(self):
        return self.df['ilae_score'].values[0]

    @property
    def clinical_difficulty(self):
        return self.df['clinical_difficulty'].values[0]

    @property
    def clinical_matching(self):
        return self.df['clinical_match'].unique()

    @property
    def clinical_center(self):
        return self.df['clinical_center'].values[0]

    @property
    def gender(self):
        return self.df['gender'].values[0]

    @property
    def age(self):
        return self.df['age_surgery'].values[0]

    @property
    def onsetage(self):
        return self.df['onset_age'].values[0]

    @property
    def handedness(self):
        return self.df['handedness'].values[0]

    @property
    def resected_contacts(self):
        return literal_eval(self.df['resected_contacts'].values[0])

    @property
    def ablated_contacts(self):
        return literal_eval(self.df['ablated_contacts'].values[0])

    @property
    def cezlobe(self):
        if 'cezlobe' in self.df.columns:
            return self.df['cezlobe'].values[0]
        else:
            return None


class DatasetClinical():
    """
    Dataset clinical metadata class for a single instance of a dataset.

    """

    def __init__(self, patid, datasetid, df):
        self.id = patid
        self.datasetid = datasetid
        self.df = df

    @property
    def patient_id(self):
        return self.df['patient_id'].values[0]

    @property
    def dataset_identifier(self):
        return self.df['dataset_identifier'].values[0]

    @property
    def clinical_seizure_identifier(self):
        return self.df['clinical_seizure_identifier'].values[0]

    @property
    def brain_location(self):
        return self.df['brain_location'].values[0]

    @property
    def clinical_semiology(self):
        return self.df['clinical_semiology'].values[0]

    @property
    def ez_hypo_contacts(self):
        # print(self.df['ez_hypo_contacts'][0])
        return self.df['ez_hypo_contacts'].values[0]

    @property
    def seizure_semiology(self):
        return self.df['seizure_semiology'].values[0]
