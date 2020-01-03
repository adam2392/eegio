import os
import tempfile
import pytest
from mne_bids import make_bids_basename
from eegio.loaders import BaseBids

from eegio.loaders.bids.bidsio import BidsWriter, BidsLoader


class TestBaseBids:
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def testbasebids(self):
        """

        Parameters
        ----------

        Returns
        -------

        """
        # check that base bids initializes correct class attributes
        with tempfile.TemporaryDirectory() as bids_root:
            basebids = BaseBids(bids_root=bids_root)
            assert os.path.exists(basebids.description_fpath)

            print(basebids.description_fpath)
            print(basebids.subject_nums)
            print(basebids.num_subjects)
            print(basebids.data_types)
            print(basebids.participants_tsv_fpath)
            print(basebids.participants_json_fpath)

            basebids.print_dir_tree()
            basebids.print_summary()

    def test_baseio(self):
        test_subjectid = "0001"
        session_id = "seizure"
        kind = "eeg"
        run_id = "01"
        task = "monitor"
        bids_basename = make_bids_basename(
            subject=test_subjectid,
            session=session_id,
            acquisition=kind,
            run=run_id,
            task=task,
            # suffix=kind + ".fif",
        )
        ext = "fif"
        bids_root = os.path.join(os.getcwd(), "./data/bids_layout/")

        # instantiate a loader/writer
        loader = BidsLoader(
            bids_root=bids_root, bids_basename=bids_basename, kind=kind, datatype=ext,
        )
        participant_dict = loader.load_participants_json()
        participant_df = loader.load_participants_tsv()
        scans_df = loader.load_scans_tsv()
        sidecar = loader.load_sidecar_json()
        chans_df = loader.load_channels_tsv()

        # check basic funcs
        print(loader.chanstsv_fpath)
        print(loader.datafile_fpath)
        print(loader.eventstsv_fpath)
        print(loader.rel_chanstsv_fpath)
        print(loader.rel_datafile_fpath)
        print(loader.rel_eventstsv_fpath)
        print(loader.rel_participantsjson_fpath)
        print(loader.rel_participantstsv_fpath)
        print(loader.rel_scanstsv_fpath)
        print(loader.rel_sidecarjson_fpath)

        with tempfile.TemporaryDirectory() as bids_root:
            writer = BidsWriter(
                bids_root=bids_root,
                bids_basename=bids_basename,
                kind=kind,
                datatype=ext,
            )
            with pytest.raises(Exception):
                writer.write_channels_tsv(chans_df)
            with pytest.raises(Exception):
                writer.write_electrode_coords()
            with pytest.raises(Exception):
                writer.write_scans_tsv(scans_df)
            with pytest.raises(Exception):
                writer.write_sidecar_json(sidecar)

            writer.write_participants_json(participant_dict)
            writer.write_participants_tsv(participant_df)
