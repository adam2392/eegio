import numpy as np
import pytest


@pytest.mark.usefixture('clinical_sheet')
class TestPatient():
    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_clinical(self, clinical_sheet):
        """
        Test to test MasterClinicalSheet object.

        :param clinical_sheet:
        :return:
        """
        pass


    def test_clinical_errors(self, clinical_sheet):
        """
        Test error and warnings raised by MasterClinicalSheet class.

        :param clinical_sheet:
        :return: None
        """
        pass
