"""
Authors: Adam Li and Patrick Myers.

Version: 1.0
"""

import os
import tempfile
from typing import Union

import mne
import mne_bids
import numpy as np
import pandas as pd
from mne_bids import make_bids_basename
from mne_bids import write_raw_bids, make_bids_folders
from mne_bids.utils import _handle_kind
from mne_bids.utils import _parse_bids_filename, _parse_ext

from eegio.base.config import BAD_MARKERS
from eegio.base.utils.scrubber import ChannelScrub


class BidsBuilder(object):
    """Suite of helper functions to build out a bids data directory. Makes folder directories and stuff"""

    @staticmethod
    def make_bids_folder_struct(datadir, subject, kind, session_id="one"):
        """
        Create the empty folder structure for a new subject according to the BIDS format.

        Session is assumed to be 'one' for now.

        Parameters
        ----------
        datadir :
        subject :
        kind :
        session_id :

        """
        if type(datadir) is not str:
            datadir = str(datadir)

        # create bids folders for this subject
        make_bids_folders(
            subject=subject,
            kind=kind,
            session=session_id,
            output_path=datadir,
            make_dir=True,
        )

    @staticmethod
    def make_preprocessed_folder(
        datadir, subject_id, session="one", preprocess_name="preprocessed"
    ):
        """
        Create the preproceesed data folder in the correct location if it does not already exist.

        Assumes session is 'one'.

        Parameters
        ----------
        datadir : Union[str, os.PathLike]
            The base directory for bids data
        subject_id : str
            The unique identifier for the BidsPatient
        session : str
            The identifier for this BidsPatient session
        preprocess_name : str
            The name of the preprocessed folder

        """
        basedir = os.path.join(
            datadir, make_bids_basename(subject=subject_id, session=session)
        )
        preprocessed_dir = os.path.join(basedir, preprocess_name)
        if os.path.exists(preprocessed_dir):
            return 1
        else:
            os.mkdir(preprocessed_dir)

    @staticmethod
    def make_results_folder(datadir, subject_id, session="one", result_name="results"):
        """
        Create the results folder in the correct location if it does not already exist.

        Assumes session is 'one'.

        Parameters
        ----------
        datadir :
        subject_id :
        session :
        result_name :

        """
        basedir = os.path.join(
            datadir, make_bids_basename(subject=subject_id, session=session)
        )
        results_dir = os.path.join(basedir, result_name)
        if os.path.exists(results_dir):
            return 1
        else:
            os.mkdir(results_dir)


class BidsConverter:
    @staticmethod
    def convert_to_bids(
        edf_fpath,
        bids_root,
        bids_basename,
        coords_fpath=None,
        excluded_contacts=None,
        eog_contacts=None,
        misc_contacts=None,
        overwrite=False,
        line_freq=60.0,
    ):
        """
        Convert the passed edf file into the Bids format.

        # TODO:
        - Clean up how write_raw_bids is called
        - eliminate redundant writing/reading using temporaryDirectory
        
        Parameters
        ----------
        edf_fpath : Union[str, os.PathLike]
            The location the edf file.
        bids_root : Union[str, os.PathLike]
            The base directory for newly created bids files.
        bids_basename : str
            The base name of the new data files
        excluded_contacts : list
            Contacts to be excluded from conversion
        eog_contacts : list
            Contacts to be annotated as EOG.
        misc_contacts : list
            Contacts to be annotated  as Misc.
        overwrite : bool
            Whether to overwrite an existing converted file.

        Returns
        -------
        The path to the new data file.

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
        if line_freq is not None:
            raw.info["line_freq"] = line_freq

        annonymize_dict = None
        # {
        #         "daysback": 10000,
        #         "keep_his": True,
        #     }

        # extract parameters from bids_basenmae
        params = _parse_bids_filename(bids_basename, True)
        subject, session = params["sub"], params["ses"]
        acquisition, kind = params["acq"], params["kind"]
        task = params["task"]

        # read in the events from the EDF file
        events_data, events_id = mne.events_from_annotations(raw)
        print(events_data, events_id)
        channel_scrub = ChannelScrub

        # convert the channel types based on acquisition if necessary
        if acquisition is not None:
            ch_modality_map = {ch: acquisition for ch in raw.ch_names}
            raw.set_channel_types(ch_modality_map)
            ch_type_mapping = channel_scrub.label_channel_types(raw.ch_names)
            raw.set_channel_types((ch_type_mapping))

        # reformat channel text if necessary
        channel_scrub.channel_text_scrub(raw)

        # look for bad channels that are obvious
        channel_names = raw.ch_names
        bad_channels = channel_scrub.look_for_bad_channels(channel_names)
        bad_channels_dict = {}
        for bad in bad_channels:
            bad_channels_dict[
                bad
            ] = f"Scrubbed channels containing markers {', '.join(BAD_MARKERS)}"
        raw.info["bads"] = bad_channels

        if coords_fpath:
            ch_pos = dict()
            with open(coords_fpath, "r") as fp:
                # strip of newline character
                lines = [line.rstrip("\n") for line in fp]

                for line in lines:
                    ch_name = line.split(" ")[0]
                    coord = line.split(" ")[1:]
                    ch_pos[ch_name] = [float(x) for x in coord]
            unit = "mm"
            if unit != "m":
                ch_pos = {
                    ch_name: np.divide(coord, 1000) for ch_name, coord in ch_pos.items()
                }
            montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")

            # TODO: remove. purely for testing scenario
            # ch_names = raw.ch_names
            # elec = np.random.random_sample((len(ch_names), 3))  # assume in mm
            # elec = elec / 1000  # convert to meters
            # montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, elec)),
            #                                         coord_frame='head')
        else:
            montage = None

        if montage is not None:
            if not isinstance(montage, mne.channels.DigMontage):
                raise TypeError(
                    "Montage passed in should be of type: " "`mne.channels.DigMontage`."
                )
            raw.set_montage(montage)
            print("Set montage: ")
            print(len(raw.info["ch_names"]))
            print(raw.info["dig"])
            print(raw)

        # actually perform write_raw bids
        bids_root = write_raw_bids(
            raw,
            bids_basename,
            bids_root,
            events_data=events_data,
            event_id=events_id,
            overwrite=overwrite,
            # anonymize=annonymize_dict,
            verbose=False,
        )

        # save a fif copy and reload it
        kind = _handle_kind(raw)
        fif_data_path = make_bids_folders(
            subject=subject,
            session=session,
            kind=kind,
            output_path=bids_root,
            overwrite=False,
            verbose=True,
        )

        bids_fname = bids_basename + f"_{kind}.fif"
        deriv_bids_root = os.path.join(bids_root, "derivatives")

        print("Should be saving for: ", bids_fname)
        with tempfile.TemporaryDirectory() as tmp_bids_root:
            raw.save(os.path.join(tmp_bids_root, bids_fname), overwrite=overwrite)
            raw = mne.io.read_raw_fif(os.path.join(tmp_bids_root, bids_fname))

            print(raw, bids_basename)
            print(raw.filenames)
            _, ext = _parse_ext(raw.filenames[0])
            print(ext)
            # actually perform write_raw bids
            bids_root = write_raw_bids(
                raw,
                bids_basename,
                deriv_bids_root,
                events_data=events_data,
                event_id=events_id,
                overwrite=overwrite,
                # anonymize=annonymize_dict,
                verbose=False,
            )
        return bids_root

    @staticmethod
    def preprocess_into_fif(bids_fname, bids_root, kind="eeg", overwrite=True):
        """
        Preprocess the edf file into fif format for easier manipulation.

        TODO:
        1. Determine if this is necessary. EDF files on one hand are hard to modify,
        but fif files have a lot of constraints.
        2. We should try to discuss w/ some folks that do this, or ask on gitter for mne_bids
        3. See: https://gist.github.com/skjerns/bc660ef59dca0dbd53f00ed38c42f6be/812cd1d4be24c0730db449ecc6eb0065da68ca51

        Parameters
        ----------
        bids_fname : Union[str, os.PathLike]
            The path to the bids format edf file
        bids_root : Union[str, os.PathLike]
            The base directory for bids data
        kind : str
            The type of data contained in the edf file.
        overwrite : bool
            Whether to overwrite an existing converted file.

        Returns
        -------
        The path of the converted fif file

        """
        params = _parse_bids_filename(bids_fname, verbose=False)
        bids_basename = make_bids_basename(
            subject=params["sub"],
            session=params["ses"],
            run=params["run"],
            suffix="eeg.edf",
        )
        raw = mne_bids.read_raw_bids(bids_basename, bids_root)

        # get events and convert to annotations
        events, events_id = mne.events_from_annotations(raw)
        onsets = events[:, 0] / raw.info["sfreq"]
        durations = np.zeros_like(onsets)  # assumes instantaneous events
        descriptions = events[:, 2]
        annotations = mne.annotations.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            orig_time=raw.info["meas_date"],
        )

        # add a bids run
        preprocessed_dir = os.path.join(
            bids_root,
            make_bids_basename(subject=params["sub"]),
            make_bids_basename(session=params["ses"]),
            f"{kind}",
        )
        if not os.path.exists(preprocessed_dir):
            os.makedirs(preprocessed_dir)

        bids_basename = make_bids_basename(
            subject=params["sub"],
            session=params["ses"],
            run=params["run"],
            processing="fif",
            suffix="raw",
        )
        # save a fif copy and reload it
        raw.save(
            os.path.join(preprocessed_dir, bids_basename + ".fif"), overwrite=overwrite
        )
        raw = mne.io.read_raw_fif(
            os.path.join(preprocessed_dir, bids_basename + ".fif")
        )

        # convert into fif
        bids_root = write_raw_bids(
            raw,
            bids_basename,
            bids_root=bids_root,
            overwrite=overwrite,
            # events_data=events,
            verbose=False,
        )
        return bids_root


class BidsUtils:
    @staticmethod
    def add_field_to_participants_json(
        current_dict: dict, new_field: str, description: Union[str, dict]
    ):
        """
        Add a new field to the participants.json file.

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
        The modified dict object

        """
        new_dict = {new_field: {"Description": description}}

        current_dict.update(new_dict)
        return current_dict

    @staticmethod
    def add_subject_to_participants(subject_id: str, loader, writer):
        """
        Add a subject into the participants.tsv file.

        Parameters
        ----------
        subject_id : str
            The name of the subject to add to the participants file

        Returns
        -------
        The modified pandas DataFrame

        """
        participants_df = loader.load_participants_tsv()
        colnames = participants_df.columns
        new_df = pd.DataFrame(
            [[subject_id] + ["n/a"] * (len(colnames) - 1)], columns=colnames
        )
        if participants_df.empty:
            participants_df = new_df
        else:
            participants_df.append(new_df)
        participants_df = participants_df.set_index("participant_id")
        writer.write_participants_tsv(participants_df)
        return participants_df

    @staticmethod
    def add_notes_to_channels(bidsrun):
        """
        Add a column to the *channels.tsv file to allow for more detailed description of status beyond good/bad.

        Parameters
        ----------
        bidsrun: eegio.BidsRun
            The object containing information about this snapshot recording.

        """
        channels_data = bidsrun.loader.load_channels_tsv()
        colnames = channels_data.columns
        if "notes" not in colnames:
            notes = [""] * channels_data.shape[0]
            channels_data["notes"] = notes
        bidsrun.writer.write_channels_tsv(channels_data)
