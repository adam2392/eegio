import os
import warnings
from typing import Union

import mne_bids
from mne_bids import make_bids_basename

from eegio.base.objects.bids.bids_patientv2 import BidsPatient
from eegio.base.utils.bids_helper import BidsBuilder
from eegio.loaders.bidsio import BidsIO, BidsWriter, BidsLoader


class BidsRun(BidsPatient):

    def __init__(
            self,
            subject_id: str,
            session_id: str,
            run_id: str,
            datadir: Union[str, os.PathLike] = None,
            modality: str="ieeg",
            original_filename: str=None,
    ):
        super(BidsRun, self).__init__(subject_id=subject_id, session_id=session_id, datadir=datadir, modality=modality)
        self.run_id = run_id
        self.original_filename = original_filename


        # instantiate a loader/writer
        self.loader = BidsLoader(datadir, subject_id, session_id, run_id, modality)
        self.writer = BidsWriter(datadir, subject_id, session_id, run_id, modality)

    @property
    def run_fname(self):
        io = BidsIO(self.bidsdir, self.subject_id, self.session_id, self.modality, self.run_id, "edf")
        return io.datafile_fpath

    def _add_original_scan_filename(self):
        run_fname = self.run_fname()
        scans_data = self.loader.load_scans_tsv()
        colnames = scans_data.columns
        if 'original_filename' not in colnames:
            fnames = []
            for index, row in scans_data.iterrows():
                fnames.append('')
            scans_data['original_filename'] = fnames
        for index, row in scans_data.iterrows():
            if row['filename'] == run_fname:
                row['original_filename'] = self.original_filename

    def _modify_electrode_status(self, runid):
        channels_data = self.loader.load_channels_tsv()
        # TODO: Parse all channels from source. Create dictionary of values to change
        # TODO: Included in dict are name as key, then type and status to change

    def _add_metadata(self, id, data):
        """
        Helper function to add patient metadata. Requires column ID and then data source to be passed in.

        Parameters
        ----------
        id :
        data :

        Returns
        -------

        """
        pass

    def _add_scandata(self, filename, acq_time, **scan_kwargs):
        pass
