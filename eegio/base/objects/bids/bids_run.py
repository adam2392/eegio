import os
from typing import Union
import pandas as pd

import warnings

from eegio.base.objects.bids.basebids import BaseBids
from eegio.loaders.bidsio import BidsIO, BidsWriter, BidsLoader

class BidsRun(BaseBids):
    """
    The class for a specific run of

    Attributes
    ----------
    patientid: str
        patient identifier for this BidsPatient
    datadir: os.PathLike
        The base directory for storing BidsPatient data
    managing_user: str
        managing user that is tied to this BidsPatient. E.g. a clinical center, specific researcher, or specific doctor
    modality: str
        Modality of recording done to this patient. E.g. ecog, seeg, scalp, ieeg.
    metadata:
        TODO

    Notes
    -----

    Examples
    --------
    """
    def __init__(
            self,
            subject_id: str,
            session_id: str,
            run_id: str,
            datadir: Union[str, os.PathLike] = None,
            modality: str = "ieeg",
            original_filename: str = None,
            original_subject_id: str = '',
    ):
        super(BidsRun, self).__init__(datadir=datadir, modality=modality)
        self.subject_id = subject_id
        self.session_id = session_id
        self.run_id = run_id
        self.original_filename = original_filename
        self.original_subject_id = original_subject_id
        self.result_fpath = None

        # instantiate a loader/writer
        self.loader = BidsLoader(datadir, subject_id, session_id, run_id, modality)
        self.writer = BidsWriter(datadir, subject_id, session_id, run_id, modality)
        self._add_original_scan_filename()

    @property
    def fpath(self):
        io = BidsIO(self.datadir, self.subject_id, self.session_id, self.modality, self.run_id, "edf")
        return io.datafile_fpath

    @property
    def results_fpath(self):
        io = BidsIO(self.datadir, self.subject_id, self.session_id, self.modality, self.run_id, "edf")
        return io.result_fpath

    def load_result(self):
        self.result_fpath = self.results_fpath

    def _add_original_scan_filename(self):
        """
        Adds the original scan filename (from clinical source) to the scans tsv folder for this BidsPatient

        Returns
        -------

        """
        scans_data = self.loader.load_scans_tsv()
        colnames = scans_data.columns
        if 'original_filename' not in colnames:
            fnames = []
            for index, row in scans_data.iterrows():
                fnames.append('')
            scans_data['original_filename'] = fnames
        for index, row in scans_data.iterrows():
            if row['filename'] == self.fpath:
                row['original_filename'] = self.original_filename

    def _modify_electrode_status(self, clinical_fpath):
        """
        Updates the run's channel information by looking at the generated clinical electrode sheet annotations

        Parameters
        ----------
        clinical_fpath : os.PathLike
            The location of the clinical formatted electrode sheet (xlsx format)

        Returns
        -------

        """
        clinical_data = pd.read_xlsx(clinical_fpath)
        channels_data = self.loader.load_channels_tsv()
        subject_data = pd.DataFrame()
        for ind, row in clinical_data.iterrows():
            if row['patient_id'] == self.subject_id or row['patient_id'] == self.original_subject_id:
                subject_data = row
        if subject_data.empty:
            warnings.warn('The patient has not been annotated in the passed clinical file')
        else:
            bad_contacts = subject_data['bad_contacts']
            wm_contacts = subject_data['wm_contacts']
            out_contacts = subject_data['out_contacts']
            modify_channels = {}
            # Will these cells be loaded as lists or a string to be parsed?
            for bad in bad_contacts:
                new_dict = {bad: {'status': 'bad', 'notes': "bad"}}
                modify_channels.update(new_dict)
            for wm in wm_contacts:
                new_dict = {wm: {'status': 'bad', 'notes': "white matter"}}
                modify_channels.update(new_dict)
            for out in out_contacts:
                new_dict = {out: {'status': 'bad', 'notes': "out"}}
                modify_channels.update(new_dict)
            channels_data = self.writer.write_electrode_status(modify_channels, channels_data)
            channels_data = self.writer.write_electrode_coords(None, channels_data)
            channels_data = self.writer.write_electrode_type(channels_data)
            self.writer.write_channels_tsv(channels_data)


    def _add_notes_to_channels(self):
        """
        Adds a column to the *channels.tsv file to allow for more detailed description of status beyond good/bad
        Returns
        -------

        """
        channels_data = self.loader.load_channels_tsv()
        colnames = channels_data.columns
        if 'Notes' not in colnames:
            notes = [''] * channels_data.shape[0]
            channels_data['Notes'] = notes
        self.writer.write_channels_tsv(channels_data)

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
