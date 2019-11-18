import os
from typing import Union

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
    ):
        super(BidsRun, self).__init__(datadir=datadir, modality=modality)
        self.subject_id = subject_id
        self.session_id = session_id
        self.run_id = run_id
        self.original_filename = original_filename

        # instantiate a loader/writer
        self.loader = BidsLoader(datadir, subject_id, session_id, run_id, modality)
        self.writer = BidsWriter(datadir, subject_id, session_id, run_id, modality)
        self._add_original_scan_filename()

    @property
    def fpath(self):
        io = BidsIO(self.datadir, self.subject_id, self.session_id, self.modality, self.run_id, "edf")
        return io.datafile_fpath

    def _add_original_scan_filename(self):
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

    def _modify_electrode_status(self):
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
