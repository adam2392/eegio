from typing import List, Dict
import json

from eegio.base.utils.data_structures_utils import NumpyEncoder
from eegio.loaders.loader import Loader
from eegio.writers.saveas import DataWriter


def run_formatting_eeg(
    in_fpath: str,
    out_fpath: str,
    json_fpath: str,
    bad_contacts: List = None,
    clinical_metadata: Dict = None,
):
    if bad_contacts is None:
        bad_contacts = []
    if clinical_metadata is None:
        clinical_metadata = dict()

    # load in the file
    loader = Loader(in_fpath, clinical_metadata)
    eegts = loader.load_file(in_fpath)

    # get the formatted fif and json files
    raw = _save_fif_file(eegts, out_fpath)
    metadata = _save_json_file(eegts, json_fpath)

    return raw, metadata


def _save_fif_file(eegts, out_fpath):
    writer = DataWriter(out_fpath)

    # get the corresponding data
    rawdata = eegts.get_data()
    info = eegts.info
    bad_chans = eegts.bad_contacts
    montage = eegts.get_montage()

    raw = writer.saveas_fif(out_fpath, rawdata, info, bad_chans, montage)
    return raw


def _save_json_file(eegts, json_fpath):
    metadata = eegts.get_metadata()

    # save the formatted metadata json object
    with open(json_fpath, "w") as fp:
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
