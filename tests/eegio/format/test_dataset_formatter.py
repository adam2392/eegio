import json
import os
import tempfile

import mne
import pandas as pd
import pytest

from eegio.format.format_clinical_sheet import FormatClinicalSheet
from eegio.format.format_eeg_data import FormatEEGData


@pytest.mark.usefixture("edf_fpath")
def test_preprocess_edf_to_fif(edf_fpath):
    with tempfile.TemporaryDirectory() as fdir:
        temp_outfpath = os.path.join(fdir, "test_raw.fif")
        temp_jsonfpath = os.path.join(fdir, "test_raw.json")
        formatter = FormatEEGData(edf_fpath, temp_outfpath, temp_jsonfpath)

        # assert saved mne file and json file can load in
        raw = mne.io.read_raw_fif(temp_outfpath, preload=True)
        with open(temp_jsonfpath, "r", encoding="utf8") as fp:
            metadata = json.load(fp)

        assert isinstance(metadata, dict)


@pytest.mark.usefixture("clinical_fpath")
def test_preprocess_clinical(clinical_fpath):
    with tempfile.TemporaryDirectory() as fdir:
        temp_outfpath = os.path.join(fdir, "test_clinical.csv")
        formatter = FormatClinicalSheet(
            clinical_fpath, cols_to_reg_expand=["bad_channels"]
        )

        # assert saved mne file and json file can load in
        formatter.write_file(temp_outfpath)

        test_csv = pd.read_csv(temp_outfpath, index_col=None)
