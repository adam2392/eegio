# -*- coding: utf-8 -*-
import re
from ast import literal_eval

import numpy as np
import pandas as pd

from eegio.base.objects.clinical import ClinicalScalpMeta, ClinicalPatientMeta, \
    ClinicalDatasetMeta
from eegio.base.utils import expand_channels


def format_list_str_channels(chanlist):
    return literal_eval(chanlist[0])


def expand_semio_channels(chanstr):
    # chanstr = format_list_str_channels(chanstr)
    # split by ->
    semiochans = chanstr.split("->")
    semiology = []
    for idx, chanlist in enumerate(semiochans):
        # perform a split based on certain delimiters
        allchansplit = re.split('; |: |,', chanlist)
        # expand these channels to the correct strings
        semiology.append(expand_channels(allchansplit))
    return semiology


class ExcelReader(object):

    def __init__(self, filepath, apply_formatting=True, expanding_semio=False):
        self.filepath = filepath

        # read in the excel pages for ieeg, datasets, scalp layout
        self.ieegdf = pd.read_excel(filepath, 'ieeg', header=0)
        self.datasetdf = pd.read_excel(filepath, 'alldatasets', header=0)
        self.scalpdf = pd.read_excel(filepath, 'scalp', header=0)

        # apply formatting changes to each dataframe read in
        if apply_formatting:
            # format these datasets
            self.datasetdf = self.format_dataset_df(self.datasetdf)
            self.ieegdf = self.format_ieeg_df(self.ieegdf)
            self.scalpdf = self.format_scalp_df(self.scalpdf)

        # expand the semiology contact labeling into a list of lists [[onset],[spread]]
        if expanding_semio:
            self.datasetdf = self.expand_semiology_contacts(self.datasetdf)

    def _all_formatting(self, clinical_df):
        # lowercase df if not already
        clinical_df = clinical_df.apply(lambda x: x.astype(str).str.lower())
        # replace "`" with "''"
        # clinical_df = clinical_df.apply(lambda x: x.replace("’", "'"))
        # map all empty to nans
        clinical_df = clinical_df.fillna(np.nan)
        clinical_df = clinical_df.replace('nan', '', regex=True)
        return clinical_df

    def read_formatted_df(self, filepath=None):
        if filepath is not None:
            self.ieegdf = ClinicalPatientMeta(
                pd.read_excel(filepath, 'ieeg', header=0))
            self.datasetdf = ClinicalDatasetMeta(
                pd.read_excel(filepath, 'alldatasets', header=0))
            self.scalpdf = ClinicalScalpMeta(
                pd.read_excel(filepath, 'scalp', header=0))
        else:
            self.ieegdf = ClinicalPatientMeta(self.ieegdf)
            self.datasetdf = ClinicalDatasetMeta(self.datasetdf)
            self.scalpdf = ClinicalScalpMeta(self.scalpdf)

        return self.ieegdf.clindf, self.datasetdf.clindf, self.scalpdf.clindf

    def merge_dfs(self, patientdf, datasetdf, scalpdf, patient_id):
        # get data for this patient
        patientdf._trimdf(patient_id=patient_id)
        datasetdf._trimdf(patient_id=patient_id)
        scalpdf._trimdf(patient_id=patient_id)

        clindf = pd.concat(
            [patientdf.clindf, datasetdf.clindf, scalpdf.clindf], sort=False)
        return clindf

    def format_dataset_df(self, clinical_df):
        # which columns to expand
        contact_cols = [
            'ez_hypo_contacts',
            # 'seizure_semiology',
        ]

        clinical_df = self._all_formatting(clinical_df)

        # do some string processing to expand out contacts
        for col in contact_cols:
            # split contacts by ";", ":", or ","
            clinical_df[col] = clinical_df[col].str.split(
                '; |: |,')  # , expand = True)
            clinical_df[col] = clinical_df[col].apply(
                lambda x: expand_channels(x))

            # print(clinical_df[col])
            # clinical_df[col] = clinical_df[col].apply(lambda x: format_list_str_channels(x))

        return clinical_df

    def format_scalp_df(self, clinical_df):
        numerical_cols = [
        ]

        lobe_cols = [
            'cezlobe'
        ]

        clinical_df = self._all_formatting(clinical_df)

        # convert numerical columns to numbers again
        for col in numerical_cols:
            clinical_df[col] = pd.to_numeric(clinical_df[col],
                                             errors='coerce')

        # do some string processing to expand out contacts
        for col in lobe_cols:
            # clinical_df[col] = clinical_df[col].map(lambda x: x.strip())
            clinical_df[col] = clinical_df[col].str.strip()
            # split contacts by ";", ":", or ","
            clinical_df[col] = clinical_df[col].str.split('; |: |,')
            clinical_df[col] = clinical_df[col].map(
                lambda x: [y.strip() for y in x])
            clinical_df[col] = clinical_df[col].map(
                lambda x: [y.replace(' ', '-') for y in x])

        return clinical_df

    def format_ieeg_df(self, clinical_df):
        numerical_cols = [
            'clinical_difficulty',
            'engel_score',
            'age_surgery',
            'onset_age',
            'clinical_match'
        ]

        # which columns to expand
        contact_cols = [
            'resected_contacts',
            'ablated_contacts',
            'bad_channels',
            'wm_contacts'
        ]

        clinical_df = self._all_formatting(clinical_df)

        # convert numerical columns to numbers again
        for col in numerical_cols:
            clinical_df[col] = pd.to_numeric(clinical_df[col],
                                             errors='coerce')

        # do some string processing to expand out contacts
        for col in contact_cols:
            # split contacts by ";", ":", or ","
            clinical_df[col] = clinical_df[col].str.split('; |: |,')
            clinical_df[col] = clinical_df[col].apply(
                lambda x: expand_channels(x))
            # clinical_df[col] = clinical_df[col].apply(lambda x: format_list_str_channels(x))

        return clinical_df

    def expand_semiology_contacts(self, clinical_df):
        clinical_df['seizure_semiology'] = clinical_df['seizure_semiology'].apply(
            lambda x: expand_semio_channels(x))

        return clinical_df

    def write_to_excel(self, outputexcelfilepath):
        writer = pd.ExcelWriter(outputexcelfilepath)
        self.ieegdf.to_excel(writer,
                             'ieeg',
                             index=None)
        self.scalpdf.to_excel(writer,
                              'scalp',
                              index=None)
        self.datasetdf.to_excel(writer,
                                'alldatasets',
                                index=None)
        writer.save()
