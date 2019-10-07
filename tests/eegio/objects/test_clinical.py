import inspect
import os
import pathlib
import tempfile

import pandas as pd
import pytest

from eegio.base.objects.clinical import PatientClinical, DatasetClinical
from eegio.loaders import DataSheetLoader


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

    def test_clinical_conversion_exceltocsv(self):
        """
        Test to test MasterClinicalSheet object.

        :param clinical_sheet:
        :return:
        """
        clinical_excel_fpath = pathlib.Path(self.clinical_excel_fpath)
        clinical_csv_fpath = pathlib.Path(self.clinical_csv_fpath)

        # instantiate datasheet loader
        clinloader = DataSheetLoader()

        # load excel file
        excel_df = clinloader.load(fpath=clinical_excel_fpath)

        # convert excel file to csv file
        with tempfile.NamedTemporaryFile() as tmp:
            print(tmp.name)
            excel_df.to_csv(tmp.name, index=None)
            temp_csv_df = clinloader.from_csv(tmp.name)

        # load csv file
        csv_df = clinloader.load(fpath=clinical_csv_fpath)

        # make sure all entries are the same
        pd.testing.assert_frame_equal(
            temp_csv_df,
            csv_df,
            check_dtype=True,
            # check_column_type=True,
            check_datetimelike_compat=True,
        )

    def test_clinical_conversion_csvtoexcel(self):
        """
        Test to test MasterClinicalSheet object.

        :param clinical_sheet:
        :return:
        """
        clinical_excel_fpath = pathlib.Path(self.clinical_excel_fpath)
        clinical_csv_fpath = pathlib.Path(self.clinical_csv_fpath)

        # instantiate datasheet loader
        clinloader = DataSheetLoader()

        # load excel file
        csv_df = clinloader.load(fpath=clinical_csv_fpath)

        # convert excel file to csv file
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.name += ".xlsx"
            print(tmp.name)
            csv_df.to_excel(tmp.name, index=None)
            temp_excel_df = clinloader.from_excel(tmp.name)

        # load csv file
        excel_df = clinloader.load(fpath=clinical_excel_fpath)

        # make sure all entries are the same
        pd.testing.assert_frame_equal(
            temp_excel_df,
            excel_df,
            check_dtype=True,
            # check_column_type=True,
            check_datetimelike_compat=True,
        )

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_clinical(self):
        """
        Test to test MasterClinicalSheet object.

        :param clinical_sheet:
        :return:
        """
        clinical_csv_fpath = pathlib.Path(self.clinical_csv_fpath)

        # instantiate datasheet loader
        clinloader = DataSheetLoader()

        # load csv file
        csv_df = clinloader.load(fpath=clinical_csv_fpath)

        # clean up the column names
        csv_df = clinloader.clean_columns(csv_df)

        """ do some stuff w/ the csv file """
        # check attributes
        attributes = inspect.getmembers(
            clinloader, lambda a: not (inspect.isroutine(a))
        )
        clin_attribs = [
            a for a in attributes if not (a[0].startswith("__") and a[0].endswith("__"))
        ]
        print(clin_attribs)
        print(csv_df.columns)
        # assert False

    def test_clinical_errors(self):
        """
        Test error and warnings raised by MasterClinicalSheet class.

        :param clinical_sheet:
        :return: None
        """
        pass
