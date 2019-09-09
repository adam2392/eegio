import os
import warnings

from eegio.loaders.dataset.ieegrecording import iEEGRecording
from eegio.loaders.dataset.scalprecording import ScalpRecording
from eegio.loaders.patient.subjectbase import SubjectBase


class SubjectRawLoader(SubjectBase):
    """
    Patient level wrapper for loading in data, or result. It helps
    load in all the datasets for a particular subject, for doing per-subject analysis.

    Assumes directory structure:
    root_dir
        -> name
            -> datasets / result_computes
    """

    def __init__(self, root_dir,
                 subjid=None,
                 preload=True,
                 datatype='ieeg',
                 reference='monopolar'):
        super(SubjectRawLoader, self).__init__(
            root_dir=root_dir, subjid=subjid, datatype=datatype)

        self.reference = reference

        # set reference for this recording and jsonfilepath
        self.modality = datatype
        self._determine_dir_structure()
        if self.datatype == 'ieeg':
            self.loadingfunc = iEEGRecording
        elif self.datatype == 'scalp':
            self.loadingfunc = ScalpRecording

        # get a list of all the jsonfilepaths possible
        self._getalljsonfilepaths(self.eegdir)

        if preload:
            self.read_all_files()

    def _determine_dir_structure(self):
        if os.path.exists(os.path.join(self.root_dir, 'seeg', 'fif')):
            self.eegdir = os.path.join(self.root_dir, 'seeg', 'fif')
        elif os.path.exists(os.path.join(self.root_dir, 'scalp', 'fif')):
            self.eegdir = os.path.join(self.root_dir, 'scalp', 'fif')
        else:
            self.eegdir = self.root_dir

    def read_all_files(self):
        if not self.is_loaded:
            # load in each file / dataset separately
            for json_file in self.jsonfilepaths:
                jsonfilepath = os.path.join(self.eegdir, json_file)

                recording = self.loadingfunc(self.root_dir,
                                             jsonfilepath,
                                             reference=self.reference,
                                             apply_mask=True,
                                             preload=False)
                eegts = recording.loadpipeline(jsonfilepath)

                # apply reference scheme change if necessary
                if self.reference == 'bipolar':
                    eegts.set_bipolar()
                elif self.reference == 'common_avg':
                    eegts.set_common_avg_ref()
                elif self.reference != 'monopolar':
                    raise ValueError(
                        "Other reference schemes besides bipolar, commonavg and monopolar have not been implemented"
                        "yet! You can pass in custom reference signal though.")

                # create data structures of the results
                self.datasets.append(eegts)
                self.dataset_ids.append(eegts.dataset_id)
        else:
            raise RuntimeError(
                "Datasets for this patient are already loaded! Need to run reset() first!")

    def get_montage(self):
        montage_list = []
        for result in self.datasets:
            metadata = result.get_metadata()
            montage = metadata['montage']
            montage_list.append(montage)

        while len(montage_list) > 0:
            newmontage = montage_list.pop()
            if newmontage != montage:
                warnings.warn(
                    "Montages are different? We will return the first one found though.")

        return montage
