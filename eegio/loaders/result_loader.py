import os
from typing import Dict, Union

import numpy as np

from eegio.base.utils.data_structures_utils import loadjsonfile
from eegio.loaders.baseloader import BaseLoader


class ResultLoader(BaseLoader):
    def __init__(self, fname: Union[str, os.PathLike] = None, metadata: Dict = None):
        super(ResultLoader, self).__init__(fname=fname)

        if metadata is None:
            metadata = {}
        self.update_metadata(**metadata)

    def load_file(self, filepath: Union[str, os.PathLike]):
        filepath = str(filepath)
        if filepath.endswith(".fif"):
            res = self.read_fif(filepath)
        elif filepath.endswith(".mat"):
            res = self.read_mat(filepath)
        elif filepath.endswith(".edf"):
            res = self.read_edf(filepath)
        else:
            raise OSError("Can't use load_file for this file extension {filepath} yet.")

        return res

    def read_NK(self, fname: Union[os.PathLike, str]):
        """
        Function to read from a Nihon-Kohden based EEG system file.

        :param fname:
        :type fname:
        :return:
        :rtype:
        """
        pass

    def read_Natus(self, fname: Union[os.PathLike, str]):
        """
        Function to read from a Natus based EEG system file.
        :param fname:
        :type fname:
        :return:
        :rtype:
        """
        pass

    def read_npzjson(
        self,
        jsonfpath: Union[str, os.PathLike],
        npzfpath: Union[os.PathLike, str] = None,
    ):
        """
        Reads a numpy stored as npz+json file combination.

        :param jsonfpath:
        :type jsonfpath:
        :param npzfpath:
        :type npzfpath:
        :return:
        :rtype:
        """
        filedir = os.path.dirname(jsonfpath)
        # load in json file
        metadata = loadjsonfile(jsonfpath)

        if npzfpath == None:
            npzfilename = metadata["resultfilename"]
            npzfpath = os.path.join(filedir, npzfilename)

        datastruct = np.load(npzfpath)
        return datastruct, metadata

    def read_npyjson(
        self,
        jsonfpath: Union[str, os.PathLike],
        npyfpath: Union[str, os.PathLike] = None,
    ):
        """
        Reads a numpy stored as npy+json file combination.

        :param jsonfpath:
        :type jsonfpath:
        :param npyfpath:
        :type npyfpath:
        :return:
        :rtype:
        """
        filedir = os.path.dirname(jsonfpath)
        # load in json file
        metadata = loadjsonfile(jsonfpath)

        if npyfpath == None:
            npyfilename = metadata["resultfilename"]
            npyfpath = os.path.join(filedir, npyfilename)

        arr = np.load(npyfpath)
        return arr, metadata
