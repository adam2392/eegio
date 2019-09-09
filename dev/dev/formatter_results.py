from eegio.format.dev.edfconversion.baseconverter import BaseConverter
from eegio.base.objects import BaseMeta
from eegio.base.utils.utils import loadjsonfile, map_to_win, compute_samplepoints

RESULTTYPES = ['ltv', 'pert', 'frag']


class ConvertResult(BaseConverter):

    def __init__(self, winsize, stepsize):
        self.winsize = winsize
        self.stepsize = stepsize

    def load_file(self, filepath):
        pass

    def load_metadata(self, metafilepath):
        metadata = loadjsonfile(metafilepath)
        return metadata

    def sync_from_raw(self, rawmetadata, type):
        # format and check metadata using static metho
        metadata = BaseMeta.format_metadata(rawmetadata)
        metadata = BaseMeta.check_metadata(metadata)

        # apply model name
        if type not in RESULTTYPES:
            raise AttributeError(
                "Type passed in must be one of {}".format(RESULTTYPES))

        rawfilename = metadata['rawfilename']
        metadata['resultfilename'] = rawfilename.replace(
            '_raw.fif', '_{}model.npz'.format(type))

        # determine sample points data struct
        numtimepoints = metadata['length_of_recording']
        samplepoints = compute_samplepoints(
            self.winsize, self.stepsize, numtimepoints)

        # map onset and offset indices to windows
        onsetind = metadata['onsetind']
        offsetind = metadata['offsetind']
        map_items = {'onsetwin': onsetind, 'offsetwin': offsetind}
        onsetwin = map_to_win(onsetind, samplepoints)
        offsetwin = map_to_win(offsetind, samplepoints)
        metadata['onsetwin'] = onsetwin
        metadata['offsetwin'] = offsetwin

        return metadata

    def convert_npz(self, newfilepath, replace=False):
        pass

    def convert_mat(self, newfilepath, replace=False):
        pass
