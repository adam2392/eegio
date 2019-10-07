import os
import pathlib

import pandas as pd
import pytest

from eegio.base.objects.clinical import PatientClinical, DatasetClinical
from eegio.loaders import DataSheetLoader

clinical_excel_fpath = os.path.join(
    os.getcwd(), "./data/clinical_examples/test_clinicaldata.xlsx"
)
clinical_csv_fpath = os.path.join(
    os.getcwd(), "./data/clinical_examples/test_clinicaldata.csv"
)
clinical_patient_excel_fpath = os.path.join(
    os.getcwd(), "./data/clinical_examples/test_clinical_singlepatient_data.xlsx"
)
clinical_snapshot_excel_fpath = os.path.join(
    os.getcwd(), "./data/clinical_examples/test_clinical_singlepatient_data.xlsx"
)

clinical_eleclayout_excel_fpath = os.path.join(
    os.getcwd(), "./data/clinical_examples/electrode_layouts/test_electrode_layout.xlsx"
)


class TestClinical:
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_clinical_objects(self):
        """
        Test to test MasterClinicalSheet object.

        :param clinical_sheet:
        :return:
        """
        pat_fpath = pathlib.Path(clinical_patient_excel_fpath)
        snapshot_fpath = pathlib.Path(clinical_snapshot_excel_fpath)

        patid = "pat01"

        loader = DataSheetLoader()

        # instantiate datasheet loader
        df = loader.load(pat_fpath)
        patloader = PatientClinical(patid=patid, df=df)

        df = loader.load(snapshot_fpath)
        snaploader = DatasetClinical(patid=patid, df=df)

        assert isinstance(patloader.df, pd.DataFrame)
        assert isinstance(snaploader.df, pd.DataFrame)
        assert patloader.summary()
        assert snaploader.summary()

    def test_datasheet_loading(self):
        fpath = pathlib.Path(clinical_eleclayout_excel_fpath)

        loader = DataSheetLoader()
        problem_contacts = loader.load_elec_layout_sheet(fpath)

        wm_contacts = problem_contacts["wm"]
        out_contacts = problem_contacts["out"]

        assert len(set(wm_contacts).intersection(out_contacts)) == 0
        assert wm_contacts
        assert out_contacts
