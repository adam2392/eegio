import os
import tempfile

import numpy as np
import pytest

from eegio.writers.saveas import TempWriter, DataWriter


@pytest.mark.usefixture("test_arr")
class TestSaveas:
    # @pytest.mark.usefixture('fif_fpath')
    # def test_saveas_edf(self, fif_fpath):
    #     """
    #     Test error and warnings raised by Result class.
    #
    #     :param result: (Result)
    #     :return: None
    #     """
    #     pass
    #
    # @pytest.mark.usefixture('edf_fpath')
    # def test_saveas_fif(self, edf_fpath):
    #     """
    #     Test error and warnings raised by Result class.
    #
    #     :param result: (Result)
    #     :return: None
    #     """
    #

    def test_saveas_npy(self, test_arr):
        """
        Test error and warnings raised by Result class.

        :param result: (Result)
        :return: None
        """
        writerobj = TempWriter()

        with tempfile.TemporaryDirectory() as fdir:
            tempfilenames = []
            for index in range(5):
                tempfilename = writerobj.save_npy_file(fdir, index=index, arr=test_arr)
                tempfilenames.append(tempfilename)

            for index, tempfilename in enumerate(tempfilenames):
                assert os.path.basename(tempfilename) == f"temp_{index}.npy"
                test_data = np.load(tempfilename)
                np.testing.assert_array_equal(test_arr, test_data)

    def test_saveas_npz(self, test_arr):
        """
        Test error and warnings raised by Result class.

        :param result: (Result)
        :return: None
        """
        pass

    def test_saveas_merge(self, test_arr):
        writerobj = TempWriter()
        with tempfile.TemporaryDirectory() as fdir:
            outputfpath = os.path.join(fdir, "merged_test.npz")

            tempfilenames = []
            for index in range(5):
                tempfilename = writerobj.save_npy_file(fdir, index=index, arr=test_arr)
                tempfilenames.append(tempfilename)

            metadata = {"patid": "hello"}
            merger = DataWriter()
            merger.merge_npy_arrays(outputfpath, tempfilenames, metadata)

            test_data = np.load(outputfpath)
            assert "result" in test_data.keys()
