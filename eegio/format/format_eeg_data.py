import json

from eegio.loaders.loader import Loader
from eegio.writers.saveas import DataWriter
from eegio.base.utils.data_structures_utils import NumpyEncoder


class FormatEEGData:
    def __init__(self, in_fpath, out_fpath, json_fpath):
        self.in_fpath = in_fpath
        self.out_fpath = out_fpath
        self.json_fpath = json_fpath

        # load in the file
        loader = Loader(in_fpath)
        eegts = loader.load_file(in_fpath)

        self.raw = self._format_fif_file(eegts)
        self.metadata = self._format_json_file(eegts)

    def __repr__(self):
        return self.raw, self.metadata

    def _format_fif_file(self, eegts):
        writer = DataWriter(self.out_fpath)

        # get the corresponding data
        rawdata = eegts.get_data()
        info = eegts.info
        bad_chans = eegts.get_bad_chs()
        montage = eegts.get_montage()

        raw = writer.saveas_fif(self.out_fpath, rawdata, info, bad_chans, montage)
        return raw

    def _format_json_file(self, eegts):
        metadata = eegts.get_metadata()

        # save the formatted metadata json object
        with open(self.json_fpath, "w") as fp:
            json.dump(
                metadata,
                fp,
                indent=4,
                sort_keys=True,
                cls=NumpyEncoder,
                separators=(",", ": "),
                ensure_ascii=False,
            )
        return metadata
