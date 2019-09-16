import os
import warnings

from typing import List, Dict


class Patient(object):
    """
    The class for our patient data that inherits a basic dataset functionality.
    The additional components for this are related to the fact that each Patient can have
    multiple EEGTimeSeries, or Results related to it.

    Attributes
    ----------
        :param patientid: patient identifier for this Patient.
        :type patientid: bool
        :param managing_user: managing user that is tied to this Patient. E.g. a clinical center, or a specific researcher, or specific doctor.
        :type managing_user: str
        :param datasetids: list of recording snapshots tied to this Patient.
        :type datasetids: List[str]
        :param modality: modality of recording done for this patient. E.g. ecog, seeg, scalp, ieeg
        :type modality: str
        :param samplerate: sampling rate for recordings. E.g. 1000 Hz
        :type samplerate: float
        :param wm_channels:
        :type wm_channels:
        :param bad_channels:
        :type bad_channels:
        :param metadata:
        :type metadata:
        :param rz_contacts:
        :type rz_contacts:
        :param soz_contacts:
        :type soz_contacts:
        :param spread_contacts:
        :type spread_contacts:
        :param brain_region_hypo:
        :type brain_region_hypo:
        :param resected_region:
        :type resected_region:
        :param lesion_present:
        :type lesion_present:

    Notes
    -----

    Examples
    --------
    """

    def __init__(
        self,
        patientid: str,
        datadir: os.PathLike,
        managing_user: str = None,
        datasetids: List[str] = [],
        modality: str = "ieeg",
        samplerate: float = None,
        wm_channels: List = [],
        bad_channels: List = [],
        metadata: Dict = dict(),
        rz_contacts: List[str] = [],
        soz_contacts: List[str] = [],
        spread_contacts: List[str] = [],
        brain_region_hypo: str = None,
        resected_region: str = None,
        lesion_present: bool = False,
    ):
        self.datadir = datadir

        # set recording metadata
        self.managing_user = managing_user
        self.samplerate = samplerate
        self.modality = modality
        self.patientid = patientid
        self.datasetids = datasetids

        # set channel metadata - for removal across all datasets
        self.bad_channels = bad_channels
        self.wm_channels = wm_channels

        # set patient - specific metadata
        self.lesion_present = lesion_present
        self.brain_region_hypo = brain_region_hypo
        self.resected_region = resected_region
        self.soz_contacts = soz_contacts
        self.spread_contacts = spread_contacts
        self.rz_contacts = rz_contacts

        # initialize other properties eventually defined
        self.metadata = metadata

    def __str__(self):
        return "{} Patient with {} Dataset Recordings " "totaling {} seconds".format(
            self.patientid, self.numdatasets, self.total_length_seconds
        )

    @classmethod
    def create_fake_example(self):
        import tempfile
        from eegio.base.objects.dataset.eegts_object import EEGTimeSeries

        eegts = EEGTimeSeries.create_fake_example()
        patientid = "testid"
        with tempfile.TemporaryDirectory() as datadir:
            patient = Patient(patientid, datadir)
        return patient

    @property
    def numdatasets(self):
        return len(self.datasetids)

    def get_filesizes(self):
        """
        Get the total filesize of all the files attached for this patient.

        :return:
        :rtype:
        """
        size = 0
        for f in self.get_all_filepaths():
            size += os.path.getsize(f)
        return size

    def summary(self):
        pass

    def pickle_results(self):
        pass

    def load_metadata(self, metadata: dict):
        self.metadata = metadata

    # @property
    # def total_length_seconds(self):
    #     return len(self.times)
    #
    # @property
    # def n_contacts(self):
    #     return len(self.contacts.chanlabels)

    def _get_full_datasetid(self, patientid, datasetid):
        """
        Function to generate a hard-coded string that represents the way we store all the dataset IDs

        :param patientid:
        :type patientid:
        :param datasetid:
        :type datasetid:
        :return:
        :rtype:
        """
        return "_".join([patientid, datasetid])

    def get_all_filepaths(self):
        """
        Generate all filepaths for this patient's dataset.

        :return:
        :rtype:
        """
        fpaths = []
        for full_id in [
            self._get_full_datasetid(self.patientid, s) for s in self.datasetids
        ]:
            fpath = os.path.join(self.datadir, full_id)
            if not os.path.exists(fpath):
                warnings.warn(
                    f"This dataset seemed to not be available for {self.patientid}: {fpath}.",
                    RuntimeWarning,
                )
            fpaths.append(fpath)
        return fpaths

    def add_bad_channels(self, bad_channels: List[str]):
        """
        Helper setter function to load additional bad_channels

        :param bad_channels:
        :type bad_channels:
        :return:
        :rtype:
        """
        self.bad_channels = bad_channels

    def add_wm_channels(self, wm_channels: List[str]):
        """
        Helper setter function to load additional wm_channels

        :param wm_channels:
        :type wm_channels:
        :return:
        :rtype:
        """
        self.wm_channels = wm_channels

    def add_datasets_to_patient(self, datasetids: List[str]):
        """
        Helper setter function to add additional dataset ids to this patient.

        :param datasetids:
        :type datasetids:
        :return:
        :rtype:
        """
        self.datasetids.extend(datasetids)
