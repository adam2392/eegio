import os
import pathlib

import pandas as pd
import pytest

from eegio.base.objects.clinical import PatientClinical, DatasetClinical
from eegio.loaders import DataSheet


class TestClinical:
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
        os.getcwd(), "./data/clinical_examples/electrode_layout.xlsx"
    )

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_clinical_objects(self):
        """
        Test to test MasterClinicalSheet object.

        :param clinical_sheet:
        :return:
        """
        pat_fpath = pathlib.Path(self.clinical_patient_excel_fpath)
        snapshot_fpath = pathlib.Path(self.clinical_snapshot_excel_fpath)

        patid = "pat01"

        loader = DataSheet()

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
        fpath = pathlib.Path(self.clinical_eleclayout_excel_fpath)

        loader = DataSheet()
        wm_contacts, out_contacts = loader.load_elec_layout_sheet(fpath)

        assert len(set(wm_contacts).intersection(out_contacts)) == 0
        assert wm_contacts
        assert out_contacts
