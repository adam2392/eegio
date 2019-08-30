# -*- coding: utf-8 -*-
from ast import literal_eval

import numpy as np
import pandas as pd
from eztrack.eegio.objects.baseobject import BaseMeta


def format_list_str_channels(chanlist):
    return literal_eval(chanlist[0])


class ClinicalPatientMeta(BaseMeta):
    """
    Class wrapper for clinical patient metadata related to the outcome, and clinical data of the patient.
    """

    def __init__(self, clindf):
        self.clindf = clindf

    def _trimdf(self, patient_id):
        self.clindf = self.clindf.loc[self.clindf['patient_id'] == patient_id]

    def get_clinical_outcome(self):
        return "{} with Engel {} and {}".format(
            self.clinical_center,
            self.engel_score,
            self.surgical_outcome
        )

    def get_clinical_difficulty(self):
        return "{} difficulty with {} matching".format(
            self.clinical_difficulty,
            self.clinical_matching
        )

    @property
    def bad_channels(self):
        return format_list_str_channels(self.clindf['bad_channels'].values())

    @property
    def outcome(self):
        return self.clindf['outcome'].unique()

    @property
    def engel_score(self):
        return self.clindf['engel_score'].unique()

    @property
    def clinical_difficulty(self):
        return self.clindf['clinical_difficulty'].unique()

    @property
    def clinical_matching(self):
        return self.clindf['clinical_match'].unique()

    @property
    def clinical_center(self):
        return self.clindf['clinical_center'].unique()

    @property
    def gender(self):
        return self.clindf['gender'].unique()

    @property
    def age(self):
        return self.clindf['age_surgery'].unique()

    @property
    def onsetage(self):
        return self.clindf['onset_age'].unique()

    @property
    def handedness(self):
        return self.clindf['handedness'].unique()

    @property
    def resected_contacts(self):
        return format_list_str_channels(self.clindf['resected_contacts'].values)

    @property
    def ablated_contacts(self):
        return format_list_str_channels(self.clindf['ablated_contacts'].values)

    def get_clinical_outcomes(self):
        pass


class ClinicalDatasetMeta(BaseMeta):
    """
    Class wrapper for the clinical dataset metadata that is unique per snapshot dataset.

    """

    def __init__(self, clindf):
        self.clindf = clindf

    def _trimdf(self, patient_id):
        self.clindf = self.clindf.loc[self.clindf['patient_id'] == patient_id]

    def trimdataset(self, dataset_id):
        # if self.clindf['dataset_id']
        self.clindf = self.clindf.loc[self.clindf['dataset_id'] == dataset_id]

    @property
    def patient_id(self):
        return self.clindf['patient_id'].values

    @property
    def dataset_identifier(self):
        return self.clindf['dataset_identifier'].values

    @property
    def clinical_seizure_identifier(self):
        return self.clindf['clinical_seizure_identifier'].values

    @property
    def brain_location(self):
        return self.clindf['brain_location'].values

    @property
    def clinical_semiology(self):
        return self.clindf['clinical_semiology'].values

    @property
    def ez_hypo_contacts(self):
        return format_list_str_channels(self.clindf['ez_hypo_contacts'].values)

    @property
    def seizure_semiology(self):
        return format_list_str_channels(self.clindf['seizure_semiology'].values)


class ClinicalScalpMeta(BaseMeta):
    """
    Class wrapper for the clinical metadata related to each scalp snapshot.

    """

    def __init__(self, clindf):
        self.clindf = clindf

    def _trimdf(self, patient_id):
        self.clindf = self.clindf.loc[self.clindf['ieeg_patient_id'] == patient_id]

    @property
    def patient_id(self):
        return self.clindf['patient_id'].values

    @property
    def cezlobe(self):
        return self.clindf['cezlobe'].values


class ClinicalMeta(BaseMeta):
    """
    Class wrapper for functionality to organize clinical metadata related to each patient.
    """

    def __init__(self, metadata):
        self.metadata = metadata

    def merge_dfs(self, patientdf, datasetdf, scalpdf, patient_id):
        # get data for this patient
        patientdf._trimdf(patient_id=patient_id)
        datasetdf._trimdf(patient_id=patient_id)
        scalpdf._trimdf(patient_id=patient_id)

        clindf = pd.concat(
            [patientdf.clindf, datasetdf.clindf, scalpdf.clindf], sort=False)
        self.clindf = clindf
        self.buffdf = clindf
        return clindf

    def reset(self):
        self.clindf = self.buffdf

    def get_patient_data(self, patient_id):
        self._trimdf(patient_id)

    def _trimdf(self, patient_id):
        self.clindf = self.clindf.loc[self.clindf['ieeg_patient_id'] == patient_id]

    def get_patient_summary(self):
        return "{hand} handed {gender} age {age} and onset at {onsetage}".format(
            hand=self.handedness,
            gender=self.gender,
            age=self.age,
            onsetage=self.onsetage
        )

    def get_clinical_hypothesis(self):
        brainstr = "{brain} region and {contact} contacts with \n".format(
            brain=self.ez_brain_hypothesis,
            contact=self.ez_contact_hypothesis,
        )
        semiostr = "onset {onset}, early spread {espread}, late spread {lspread} \n".format(
            onset=self.onset_brain,
            espread=self.early_spread_brain,
            lspread=self.late_spread_brain
        )
        return brainstr + semiostr

    def get_clinical_outcome(self):
        return "{} with Engel {} and {}".format(
            self.clinical_center,
            self.engel_score,
            self.surgical_outcome
        )

    def get_clinical_difficulty(self):
        return "{} difficulty with {} matching".format(
            self.clinical_difficulty,
            self.clinical_matching
        )

    def augment_metadata(self, clindf):
        datacols = [
            'resected_contacts',
            'ablated_contacts',
            'clinical_difficulty',
            'outcome',
            'engel_score',
            'age_surgery',
            'onset_age',
            'clinical_match',
            'clinical_center',
            'gender',
            'cezlobe',
            'handedness',
            'ez_hypo_brainregion',
            'clinical_semiology',
            'ez_hypo_contacts',
            'seizure_semiology',
            'bad_channels',
            'wm_contacts',
            'modality',
            'ilae_score',
        ]

        # which columns to expand and expand literally
        transform_literal = [
            'resected_contacts',
            'ablated_contacts',
            'seizure_semiology',
            'ez_hypo_contacts',
            'onset_contacts',
            'cezlobe',
            'bad_channels',
            'wm_contacts'
        ]

        leftover = []

        print("\n\n These are the clinical df column values: ")
        print(list(clindf.columns.values))
        print("\n\n")

        # loop through all data columns we want to get
        for key in datacols:
            # if we found one that's in this dataframe passe din
            if key in list(clindf.columns.values):
                # get dataframe key value
                clindfvalue = clindf[key].values

                if key == 'seizure_semiology':
                    print("Semio: ",  clindfvalue)

                # do we need to literal transform the string?
                if key in transform_literal and clindfvalue:
                    clindfvalue = format_list_str_channels(clindfvalue)
                elif len(clindfvalue) == 1:
                    clindfvalue = clindfvalue[0]

                if key == 'bad_channels' and key in self.metadata.keys():
                    print("Resetting bad channels! ")
                    print(self.metadata['bad_channels'])
                    print(clindfvalue)
                    self.metadata['bad_channels'].extend(clindfvalue)
                else:
                    # set metadata with that value
                    self.metadata[key] = clindfvalue

                if key == 'ez_hypo_contacts':
                    print("EZ Hypo Contacts: ", self.metadata[key], clindfvalue)
            else:
                leftover.append(key)
        if datacols:
            print("Left over data points for: ", leftover)

    @property
    def onset_brain(self):
        return format_list_str_channels(self.clindf['clinicalez_lobe'].values)

    @property
    def onset_contact(self):
        pass

    @property
    def early_spread_brain(self):
        pass

    @property
    def late_spread_brain(self):
        pass

    @property
    def early_spread_contact(self):
        pass

    @property
    def late_spread_contact(self):
        pass

    def augmentablationcontacts(self, chanlabels):
        """
        Helper method to get additional contacts caused by ablation.

        Assumes that ablations takes out +/- contacts from ablation site.

        :param chanlabels:
        :param onsetchans:
        :return:
        """
        # get the channels that were clinically annotated
        onsetchans = self.ez_contact_hypothesis

        # get list of tuples of all contacts with electrode name + number
        contacts = []
        for seeg_contact in chanlabels:
            thiscontactset = False
            for idx, s in enumerate(seeg_contact):
                if s.isdigit() and not thiscontactset:
                    elec_label = seeg_contact[0:idx]
                    thiscontactset = True
            contacts.append((elec_label, int(seeg_contact[len(elec_label):])))

        # create a list of additional contacts that would be "ablated"
        additional_contacts = []
        for jdx, chan in enumerate(onsetchans):
            # split into number and contact
            thiscontactset = False
            for idx, s in enumerate(chan):
                if s.isdigit() and not thiscontactset:
                    elec_label = chan[0:idx]
                    thiscontactset = True
            channum = int(chan[len(elec_label):])

            # get the max label possible in the schema for this electrode
            allcontacts = np.array(contacts)
            contactinds = np.where(allcontacts[:, 0] == elec_label)[0]
            labelmax = np.max(allcontacts[contactinds, 1].astype(int))

            # add the ablation range contacts +/- 2 contacts
            lowerrange = np.max([channum - 2, 1])
            upperrange = np.min([channum + 3, labelmax + 1])
            ablationrange = np.arange(lowerrange, upperrange, dtype='int')
            for num in ablationrange:
                if elec_label + str(num) in chanlabels:
                    additional_contacts.append(elec_label + str(num))

        # new onsetchans
        onsetchans = np.sort(list(set(additional_contacts) - set(onsetchans)))
        return onsetchans
