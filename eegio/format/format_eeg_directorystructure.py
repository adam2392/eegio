import datetime
import json
import os
from typing import List, Dict

from eegio.base.utils.data_structures_utils import NumpyEncoder
from eegio.loaders.loader import Loader
from eegio.writers.saveas import DataWriter
from mne_bids import write_raw_bids, make_bids_basename
from mne_bids.utils import print_dir_tree

def run_formatting_eeg_directory(
        datadir: str,
    study_name: str="database",
    modality: str="eeg",
    subjects: List[str]=[],
):
    # create study path
    study_path = os.path.join(datadir, study_name)
    if not os.path.exists(study_path):
        os.makedirs(study_path)

    # Now convert our data to be in a new BIDS dataset.
    bids_basename = make_bids_basename(subject=subject_id, task=task)
    write_raw_bids(raw_file, bids_basename, output_path, event_id=trial_type,
                   events_data=events, overwrite=True)


    return 1