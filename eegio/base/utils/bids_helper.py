import os

from typing import Union
import pandas as pd

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
        if type(datadir) is not str:
            datadir = str(datadir)

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

        Parameters
        ----------
        edf_fpath :
        bids_dir :
        bids_basename :
        excluded_contacts :
        eog_contacts :
        misc_contacts :
        overwrite :

        Returns
        -------

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
    def make_preprocessed_folder(datadir, subject_id, session='one', preprocess_name='preprocessed'):
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
        basedir = os.path.join(datadir, 'sub-' + subject_id, 'ses-' + session)
        preprocessed_dir = os.path.join(basedir, preprocess_name)
        if os.path.exists(preprocessed_dir):
            return 1
        else:
            os.mkdir(preprocessed_dir)

    @staticmethod
    def make_results_folder(datadir, patientid, session='one', result_name='results'):
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
        basedir = os.path.join(datadir, 'sub-' + patientid, 'ses-'+session)
        results_dir = os.path.join(basedir, result_name)
        if os.path.exists(results_dir):
            return 1
        else:
            os.mkdir(results_dir)

    @staticmethod
    def add_field_to_dict(current_dict: dict, new_field: str, description: Union[str, dict]):
        """
        Adds a new field to the participants.json file

        Parameters
        ----------
        current_dict : dict
            The dictionary to add to
        new_field : str
            The name of the new field
        description: Union[str, dict]
            The description of the new field
        Returns
        -------
        The modified dict variable
        """
        new_dict = {new_field: description}

        current_dict.update(new_dict)
        return current_dict

    @staticmethod
    def add_subject_to_participants(subject_id: str, loader, writer):
        """

        Parameters
        ----------
        subject_id : str
            The name of the subject to add to the participants file

        Returns
        -------

        """
        participants_df = loader.load_participants_tsv()
        colnames = participants_df.columns
        new_df = pd.DataFrame([[subject_id]+['n/a']*(len(colnames)-1)], columns=colnames)
        if participants_df.empty:
            participants_df = new_df
        else:
            participants_df.append(new_df)
        participants_df = participants_df.set_index('participant_id')
        writer.write_participants_tsv(participants_df)

    def create_channel_info(self, bidsrun):
        """
        TODO:
        - This shouldn't be here. A user should ideally just have done this when setting up the BIDS folder, and
        dataset.
        -

        This method searches for automatic patterns in the electrode names to modify the channels.tsv file

        Returns
        -------
        """
        self.add_notes_to_channels(bidsrun)
        channel_data = bidsrun.loader.load_channels_tsv()
        bad_channel_dict = bidsrun.find_bad_channels()
        for key, value in bad_channel_dict.items():
            channel_data = bidsrun.modify_electrode_status(channel_data, key, value)
        channel_types_dict = bidsrun.find_channel_types()
        for key, value in channel_types_dict.items():
            channel_data = bidsrun.modify_electrode_type(channel_data, key, value)
        bidsrun.writer.write_channels_tsv(channel_data)

    @staticmethod
    def add_notes_to_channels(bidsrun):
        """
        Adds a column to the *channels.tsv file to allow for more detailed description of status beyond good/bad

        Returns
        -------

        """
        channels_data = bidsrun.loader.load_channels_tsv()
        colnames = channels_data.columns
        if 'notes' not in colnames:
            notes = [''] * channels_data.shape[0]
            channels_data['notes'] = notes
        bidsrun.writer.write_channels_tsv(channels_data)