import os

import mne
from mne_bids import write_raw_bids, make_bids_folders


class BidsBuilder(object):
    """
    Suite of helper functions to build out a bids data directory.

    """
    @staticmethod
    def make_bids_folder_struct(datadir, subject, modality, session_id="one"):
        """
        Creates the empty folder structure for a new subject according to the BIDS format.
        Session is assumed to be 'one' for now.

        Parameters
        ----------
        datadir :
        subject :
        modality :
        session_id :

        Returns
        -------

        """
        # create bids folders for this subject
        make_bids_folders(subject=subject,
                          kind=modality,
                          session=session_id,
                          output_path=datadir,
                          make_dir=True)

    @staticmethod
    def convert_to_bids(edf_fpath, bids_dir, bids_basename,
                        excluded_contacts=None, eog_contacts=None, misc_contacts=None, overwrite=False):
        """
        Helper function to convert an edf file into bids format and create all
        necessary metadata files corresponding to this data from Bids format.

        Parameters:
        -----------
        data: os.PathLike
            The path of the edf file to be loaded
        datadir:

        savePath: os.PathLike
            The path to save the resulting edf file. Named with Bids convention
        exluded_contacts: List[str]
            The list of contacts to exclude from conversion
        """
        if excluded_contacts is None:
            excluded_contacts = ["-", ""]
        raw = mne.io.read_raw_edf(
            edf_fpath,
            preload=False,
            verbose="ERROR",
            exclude=excluded_contacts,
            eog=eog_contacts,
            misc=misc_contacts,
        )
        output_path = write_raw_bids(raw, bids_basename,
                                     output_path=bids_dir,
                                     overwrite=overwrite,
                                     verbose=False)
        return output_path

    @staticmethod
    def make_preprocessed_folder(datadir, subject_id, session='ses-one', preprocess_name='preprocessed'):
        """
        Creates the preproceesed data folder in the correct location if it does
        not already exist. Assumes session is 'one'.

        Parameters
        ----------
        datadir :
        subject_id :
        session :
        preprocess_name :

        Returns
        -------

        """
        basedir = os.path.join(datadir, 'sub-' + subject_id, session)
        preprocessed_dir = os.path.join(basedir, preprocess_name)
        if os.path.exists(preprocessed_dir):
            return 1
        else:
            os.mkdir(preprocessed_dir)

    @staticmethod
    def make_results_folder(datadir, patientid, session='ses-one', result_name='results'):
        """
        Creates the results folder in the correct location if it does not already
        exist. Assumes session is 'one'.

        Parameters
        ----------
        datadir :
        patientid :
        session :
        result_name :

        Returns
        -------

        """
        basedir = os.path.join(datadir, 'sub-' + patientid, session)
        results_dir = os.path.join(basedir, result_name)
        if os.path.exists(results_dir):
            return 1
        else:
            os.mkdir(results_dir)

    @staticmethod
    def add_participant_field(current_dict, new_field, labels=None, descriptors=None):
        """
        Adds a new field to the participants.json file

        Parameters
        ----------
        current_dict : dict
            The participants dictionary to add to
        new_field : str
            The name of the new field
        labels : List[str]
            The name of subfields for the new field.
        descriptors: List[str]
            The values for each label. If labels empty, corresponds to the description of the new_field
        Returns
        -------
        The modified dict variable
        """
        if type(descriptors) is not list:
            descriptors = [descriptors]
        if labels is None:
            if len(descriptors) != 1:
                pass
            else:
                new_dict = {new_field: descriptors[0]}
        else:
            new_dict = {new_field: {}}
            values = tuple(zip(labels, descriptors))
            for lab, desc in values:
                print(lab, desc)
                new_dict[new_field][lab] = desc

        current_dict.update(new_dict)
        return current_dict
