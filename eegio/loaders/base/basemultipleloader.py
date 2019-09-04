import itertools
import os
import warnings

import numpy as np

from eegio.loaders.base.baseio import BaseIO
from base.utils.data_structures_utils import ensure_list


class BaseMultipleDatasetLoader(BaseIO):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        # set the sub directory files
        self.eegdir = os.path.join(self.root_dir, 'seeg', 'fif')

    def get_results(self):
        """
        Function to return all loaded dataset objects.

        :return: datasets (list) a list of all the datasets that were loaded
        """
        return self.datasets

    def _getall_subjsonfilepaths(self):
        """
        Helper function to get all the jsonfile names for
        user to choose one. Returns them in natural sorted order.

        Gets:
         - all file endings with .json extension
         - without '.' as file pretext

        :return: None
        """

        self.jsonfilepaths = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.json') \
                        and not file.startswith('.'):
                    self.jsonfilepaths.append(os.path.join(root, file))

    def _getallsubjectnames(self):
        """
        Helper function to get all the subject names inside the jsonfilepaths that we can load from.

        :return: None
        """

        for jsonfilepath in self.jsonfilepaths:
            jsonfilename = os.path.basename(jsonfilepath)

            patid = jsonfilename.split('_')[0]
            self.subjids.append(patid)
        self.subjids = np.unique(self.subjids)

    def _get_corresponding_npzfile(self, jsonfile):
        return jsonfile.replace("frag.json", "fragmodel.npz")

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
            subjectdatasets = subjloader.subjectdatasets
            dataset_ids = subjloader.dataset_ids
            outcome = subjloader.get_outcomes(returnlist=True)
            engel_score = subjloader.get_engelscores(returnlist=True)
            clindiff = subjloader.get_clinicaldifficulty(returnlist=True)

            centerdatasets.extend(subjectdatasets)

            clindifficulty_list.extend(clindiff)
            outcome_list.extend(outcome)
            engelscore_list.extend(engel_score)
            centerdatasetids.extend(dataset_ids)
            centerpatientids.extend(
                list(itertools.repeat(subjid, len(dataset_ids))))

        print(engelscore_list)
        print(clindifficulty_list)

        self.datasets = np.array(centerdatasets)
        self.datasetids = np.array(centerdatasetids)
        self.patientids = np.array(centerpatientids)
        self.engelscore_list = np.array(engelscore_list)
        self.clindifficulty_list = np.array(clindifficulty_list)
        self.outcome_list = np.array(outcome_list)

        return centerdatasets, centerpatientids, centerdatasetids

    def split_cez_oez(self, **kwargs):
        # unload and split into oez vs cez
        oez_mat = []
        cez_mat = []
        for idx, dataset in enumerate(self.datasets):
            print(dataset)
            clinonset_map, others_map = dataset.format_dataset(dataset,
                                                               **kwargs
                                                               # apply_trim=True,
                                                               # apply_thresh=False,
                                                               # is_scalp=False
                                                               )
            if idx == 0:
                print(clinonset_map.shape, others_map.shape)
            cez_mat.append(clinonset_map)
            oez_mat.append(others_map)
        cez_mat = np.array(cez_mat)
        oez_mat = np.array(oez_mat)
        return cez_mat, oez_mat

    def split_by_hemisphere(self, offset_sec=10, threshlevel=0.7):
        oez_mat = []
        cez_mat = []

        all_hemispheres = []

        for idx, dataset in enumerate(self.datasets):
            dataset.reset()
            # print(dataset)
            dataset.compute_montage_groups()
            hemisphere, inds = dataset.get_cez_hemisphere()
            otherinds = [ind for ind in range(
                dataset.ncontacts) if ind not in inds]

            # partition into cir and oir
            clinonset_map = dataset.get_data()[inds]
            others_map = dataset.get_data()[otherinds]

            # trim dataset in time
            clinonset_map = dataset.trim_aroundseizure(
                offset_sec=offset_sec, mat=clinonset_map)
            others_map = dataset.trim_aroundseizure(
                offset_sec=offset_sec, mat=others_map)

            # apply thresholding
            clinonset_map = dataset.apply_thresholding_smoothing(
                threshold=threshlevel, mat=clinonset_map)
            others_map = dataset.apply_thresholding_smoothing(
                threshold=threshlevel, mat=others_map)

            clinonset_map = np.nanmean(clinonset_map, axis=0)
            others_map = np.nanmean(others_map, axis=0)

            if idx == 0:
                print(clinonset_map.shape, others_map.shape)

            all_hemispheres.append(hemisphere)
            cez_mat.append(clinonset_map)
            oez_mat.append(others_map)

        cez_mat = np.array(cez_mat)
        oez_mat = np.array(oez_mat)
        return cez_mat, oez_mat, all_hemispheres

    def split_by_quadrants(self, offset_sec=10, threshlevel=0.7):
        oez_mat = []
        cez_mat = []

        all_quadrants = []

        for idx, dataset in enumerate(self.datasets):
            dataset.reset()
            dataset.compute_montage_groups()
            quadrants, inds = dataset.get_cez_quadrant()
            otherinds = [ind for ind in range(
                dataset.ncontacts) if ind not in inds]
            # partition into cir and oir
            clinonset_map = dataset.get_data()[inds]
            others_map = dataset.get_data()[otherinds]

            # trim dataset in time
            clinonset_map = dataset.trim_aroundseizure(
                offset_sec=offset_sec, mat=clinonset_map)
            others_map = dataset.trim_aroundseizure(
                offset_sec=offset_sec, mat=others_map)

            # apply thresholding
            clinonset_map = dataset.apply_thresholding_smoothing(
                threshold=threshlevel, mat=clinonset_map)
            others_map = dataset.apply_thresholding_smoothing(
                threshold=threshlevel, mat=others_map)

            clinonset_map = np.nanmean(clinonset_map, axis=0)
            others_map = np.nanmean(others_map, axis=0)

            if idx == 0:
                print(clinonset_map.shape, others_map.shape)

            all_quadrants.append(quadrants)
            cez_mat.append(clinonset_map)
            oez_mat.append(others_map)

        cez_mat = np.array(cez_mat)
        oez_mat = np.array(oez_mat)
        return cez_mat, oez_mat, all_quadrants

    def split_engelscore(self):
        engels = np.arange(1, 5).astype(int)
        engelscore_dict = dict()

        for engelscore in engels:
            engelinds = self.get_engelinds(engelscore)
            engelscore_dict[engelscore] = self.datasets[engelinds]

        return engelscore_dict

    def split_outcomes(self):
        success_inds = self.success_inds
        fail_inds = self.fail_inds

        success_dict = {
            'datasets': self.datasets[success_inds],
            'datasetids': self.datasetids[success_inds],
            'patientids': self.patientids[success_inds]
        }
        fail_dict = {
            'datasets': self.datasets[fail_inds],
            'datasetids': self.datasetids[fail_inds],
            'patientids': self.patientids[fail_inds]
        }
        return success_dict, fail_dict

    def split_clinicaldifficulty(self):
        clindiff = np.arange(1, 5).astype(int)
        clindiff_dict = {}

        for diff in clindiff:
            diffinds = self.get_difficultyinds(diff)
            clindiff_dict[diff] = self.datasets[diffinds]

        return clindiff_dict

    @property
    def success_inds(self):
        return [idx for idx, outcome in enumerate(self.outcome_list) if outcome == 's']

    @property
    def fail_inds(self):
        return [idx for idx, outcome in enumerate(self.outcome_list) if outcome == 'f']

    @property
    def engel1_inds(self):
        return [idx for idx, score in enumerate(self.engelscore_list) if score == 1]

    def get_engelinds(self, engelscore):
        return [idx for idx, score in enumerate(self.engelscore_list) if score == engelscore]

    def get_difficultyinds(self, clinicaldiff):
        return [idx for idx, diff in enumerate(self.clindifficulty_list) if diff == clinicaldiff]

    def get_patients(self):
        patients_list = []
        for result in self.datasets:
            metadata = result.get_metadata()
            patient_id = metadata['patient_id']
            patients_list.append(patient_id)
        return patients_list

    def get_datasets(self):
        dataset_list = []
        for result in self.datasets:
            metadata = result.get_metadata()
            dataset_id = metadata['dataset_id']
            dataset_list.append(dataset_id)
        return dataset_list

    def get_clinical_onsetcontacts(self):
        onsetcontacts_list = []
        for result in self.datasets:
            metadata = result.get_metadata()
            clincontacts = metadata['clinezelecs']
            onsetcontacts_list.append(clincontacts)

        return onsetcontacts_list

    def get_clinical_onsetinds(self):
        onsetcontacts_list = []
        for result in self.datasets:
            metadata = result.get_metadata()
            clincontacts = metadata['clinezelecs']
            onsetcontacts_list.append(clincontacts)

        return onsetcontacts_list

    def get_channel_labels(self):
        contacts_list = []
        for result in self.datasets:
            metadata = result.get_metadata()
            contacts = metadata['chanlabels']
            contacts_list.append(contacts)

        return contacts_list

    def get_outcomes(self, returnlist=False):
        outcomes_list = []
        for result in self.datasets:
            metadata = result.get_metadata()
            outcome = metadata['outcome']
            outcomes_list.append(outcome)

        # print(self.datasets)
        # print(outcomes_list)
        outcome = self._ensure_unique_attribute(outcomes_list, name='outcome')

        if returnlist:
            return outcomes_list
        else:
            return outcome

    def get_engelscores(self, returnlist=False):
        engelscore_list = []
        for result in self.datasets:
            metadata = result.get_metadata()
            outcome = metadata['engel_score']
            engelscore_list.append(outcome)

        engelscore = self._ensure_unique_attribute(
            engelscore_list, name='engelscore')

        if returnlist:
            return engelscore_list
        else:
            return engelscore

    def get_clinicaldifficulty(self, returnlist=False):
        clinicaldifficulty_list = []
        for result in self.datasets:
            metadata = result.get_metadata()
            clindiff = metadata['clinical_difficulty']
            clinicaldifficulty_list.append(clindiff)

        clindiff = self._ensure_unique_attribute(
            clinicaldifficulty_list, name='clinical_difficulty')
        if returnlist:
            return clinicaldifficulty_list
        else:
            return clindiff

    def get_samplerate(self):
        samplerate_list = []
        for result in self.datasets:
            metadata = result.get_metadata()
            samplerate = metadata['samplerate']
            samplerate_list.append(samplerate)

        samplerate = self._ensure_unique_attribute(
            samplerate_list, name='samplerate')
        return samplerate

    def _ensure_unique_attribute(self, attriblist, name=None):
        attriblist = ensure_list(attriblist)

        idx = 0
        attrib = attriblist[idx]
        while idx < len(attriblist):
            newattrib = attriblist[idx]
            if attrib != newattrib:
                warnings.warn(
                    "All attributes {} should be the same here!".format(name))
            idx += 1
        return attrib
