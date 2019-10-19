import os
import pathlib
import warnings

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

clinical_eleclayout_excel_fpath_normal = os.path.join(
    os.getcwd(), "./data/clinical_examples/electrode_layouts/test_electrode_layout.xlsx"
)

clinical_eleclayout_excel_fpath_formatted = os.path.join(
    os.getcwd(),
    "./data/clinical_examples/electrode_layouts/test_electrode_layout_2.xlsx",
)

clinical_eleclayout_excel_fpath_empty_edge = os.path.join(
    os.getcwd(),
    "./data/clinical_examples/electrode_layouts/test_electrode_layout_empty_edge.xlsx",
)

clinical_eleclayout_excel_fpath_empty_center = os.path.join(
    os.getcwd(),
    "./data/clinical_examples/electrode_layouts/test_electrode_layout_empty_center.xlsx",
)


class TestClinical:
    """@pytest.mark.filterwarnings('ignore:RunTimeWarning')"""

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
        fpath = pathlib.Path(clinical_eleclayout_excel_fpath_normal)

        loader = DataSheetLoader()
        problem_contacts = loader.load_elec_layout_sheet(fpath)

        wm_contacts = problem_contacts["wm"]
        out_contacts = problem_contacts["out"]

        assert len(set(wm_contacts).intersection(out_contacts)) == 0
        assert wm_contacts
        assert out_contacts

    @pytest.mark.parametrize(
        "test_params",
        [
            pytest.param(
                {  # Full sheet with no formatting
                    "elec_layout_fpath": clinical_eleclayout_excel_fpath_normal,
                    "expected_warnings": 0,
                    "warning_message": None,
                },
                marks=pytest.mark.xfail,
            ),
            pytest.param(
                {  # Full sheet with formatting
                    "elec_layout_fpath": clinical_eleclayout_excel_fpath_formatted,
                    "expected_warnings": 0,
                    "warning_message": None,
                },
                marks=pytest.mark.xfail,
            ),
            {  # Empty cells on edge of table
                "elec_layout_fpath": clinical_eleclayout_excel_fpath_empty_edge,
                "expected_warnings": 1,
                "warning_message": "Please fill in the contacts ['P’16', 'P’15', 'P’3', 'P’2', 'P’1', 'R’16', 'M’16', 'N’16', 'F’16']",
            },
            {
                "elec_layout_fpath": clinical_eleclayout_excel_fpath_empty_center,
                "expected_warnings": 1,
                "warning_message": "Please fill in the contacts ['F’12', 'F’11', 'F’10', 'F’9', 'G’9', 'I’9', 'G9', 'O’9', 'M9', 'N9']",
            },
        ],
    )
    def test_empty_contacts(self, test_params):
        fpath = pathlib.Path(test_params["elec_layout_fpath"])

        loader = DataSheetLoader()
        with pytest.warns(UserWarning) as record:
            problem_contacts = loader.load_elec_layout_sheet(fpath)
        # assert len(record) == 1
        # assert record[0].message.args[0] == test_params["warning_message"]
