import os

from natsort import natsorted

from base.utils.data_structures_utils import ensure_list
from eegio.loaders.base.baseio import BaseIO
from eegio.base.utils.utils import load_szinds
from eegio.base.utils.utils import merge_metadata


class BaseResultsLoader(BaseIO):
    """
    Base class for any algorithm results data loader from eeg time series data (i.e. scalp EEG, SEEG, ECoG).

    Attributes
    ----------
    jsonfilepath: os.PathLike
        The filepath that leads to the accompanying .json file for each resulting analysis.
    results_dir : os.PathLike
        The root directory that datasets will be located.
    result : object
        The result data object to be returned

    Notes
    -----
    """

    def __init__(self, jsonfilepath, results_dir):
        # initialize root directory for result files, result object and loading flag
        self.results_dir = results_dir
        self.result = None
        self.is_loaded = False

        # initialize empty data structure
        self.jsonfilepath = jsonfilepath
        if self.jsonfilepath is None:
            self._getalljsonfilepaths()

    def __str__(self):
        return "Result loader of {}".format(self.result)

    @property
    def samplerate(self):
        return self.metadata['samplerate']

    @property
    def cezcontacts(self):
        return self.metadata['ez_hypo_contacts']

    @property
    def resectedcontacts(self):
        return self.metadata['resected_contacts']

    @property
    def channel_semiology(self):
        return self.metadata['semiology']

    @property
    def cezlobe(self):
        return self.metadata['cezlobe']

    @property
    def record_filename(self):
        return self.metadata['filename']

    @property
    def bad_channels(self):
        return self.metadata['bad_channels']

    @property
    def non_eeg_channels(self):
        return self.metadata['non_eeg_channels']

    @property
    def patient_id(self):
        return self.metadata['patient_id']

    @property
    def dataset_id(self):
        return self.metadata['dataset_id']

    @property
    def meas_data(self):
        return self.metadata['date_of_recording']

    @property
    def length_of_recording(self):
        return self.metadata['length_of_recording']

    @property
    def numberchans(self):
        return self.metadata['number_chans']

    @property
    def clinical_center(self):
        return self.metadata['clinical_center']

    @property
    def dataset_events(self):
        return self.metadata['events']

    @property
    def onsetind(self):
        if 'onsetind' in self.metadata.keys():
            return self.metadata['onsetind']
        else:
            return None

    @property
    def offsetind(self):
        if 'offsetind' in self.metadata.keys():
            return self.metadata['offsetind']
        else:
            return None

    @property
    def onsetsec(self):
        return self.metadata['onset']

    @property
    def offsetsec(self):
        return self.metadata['termination']

    @property
    def resultfilename(self):
        return self.metadata['resultfilename']

    @property
    def chanlabels(self):
        if 'chanlabels' in self.metadata.keys():
            return self.metadata['chanlabels']
        else:
            return None

    @property
    def samplepoints(self):
        return self.metadata['samplepoints']

    @property
    def timepoints(self):
        return self.metadata['timepoints']

    def reset(self):
        """
        Resetting method to reset loader to initial variables
        :return:
        """
        self.is_loaded = False

    def load_metadata(self, metadata):
        """
        Function to load a dictionary metadata object into our result loader, so that we have access directly to some
        important metadat, such as:
            - type
            - onset window
            - offset window

        :param metadata:
        :return:
        """
        self.metadata = metadata
        # extract type information and other channel information
        if 'type' in metadata.keys():
            self.type = metadata['type']
        else:
            self.type = None

        # extract patient id
        if 'note' in metadata.keys():
            self.note = metadata['note']
        else:
            self.note = None

        onsetwin, offsetwin = load_szinds(
            self.onsetind, self.offsetind, self.samplepoints)
        try:
            self.onsetwin = ensure_list(onsetwin)[0]
        except:
            self.onsetwin = None

        try:
            self.offsetwin = ensure_list(offsetwin)[0]
        except:
            self.offsetwin = None

        self.metadata['onsetwin'] = self.onsetwin
        self.metadata['offsetwin'] = self.offsetwin

    def _getalljsonfilepaths(self, patid=None):
        """
        Helper function to get all the jsonfile names for
        user to choose one. Returns them in natural sorted order.

        Gets:
         - all file endings with .json extension
         - without '.' as file pretext
        :param patid: (optional; str) a patient identifier to only get jsonfilepaths with the <patid>_ inside the filename.
        :return: None
        """
        jsonfilepaths = [f for f in os.listdir(self.results_dir)
                         if f.endswith('.json')
                         if not f.startswith('.')]
        if patid is not None:
            jsonfilepaths = [f for f in jsonfilepaths if patid + '_' in f]
        self.jsonfilepaths = natsorted(jsonfilepaths)

    def update_metadata(self, metadata):
        """
        Function to update metadata from this result using a passed in metadata object.

        Note that you are essentially performing a merging of dictionaries! So be careful
        what you pass in.

        :param metadata: (dict) the metadata for this result timeseries.
        :return: metadata (dict) the updated metadata data structure.
        """
        # perform updates on metadata json object
        metadata = merge_metadata(metadata, self.metadata, overwrite=True)
        metadata['onsetwin'] = self.onsetwin
        metadata['offsetwin'] = self.offsetwin
        metadata['resultfilename'] = self.resultfilename
        self.metadata = metadata
        return metadata
