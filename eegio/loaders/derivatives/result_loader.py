import os
from typing import Dict, Union

import numpy as np

from eegio.base.objects import Result, Contacts
from eegio.base.utils.data_structures_utils import loadjsonfile
from eegio.loaders.derivatives.baseloader import BaseLoader


class ResultLoader(BaseLoader):
    def __init__(self, fname: Union[str, os.PathLike] = None, metadata: Dict = None):
        super(ResultLoader, self).__init__(fname=fname)

        if metadata is None:
            metadata = {}
        self.update_metadata(**metadata)

    def _wrap_result_in_obj(self, datastruct, metadata):
        """
        Helps wrap a dictionary returned result data in the form of np.ndarrays, along with
        corresponding metadata into a Result object.

        Parameters
        ----------
        datastruct :
        metadata :

        Returns
        -------

        """
        # ensure data quality
        pertmats = datastruct["pertmats"]
        delvecs = datastruct["delvecs"]
        adjmats = datastruct["adjmats"]
        chlabels = metadata["chanlabels"]

        assert adjmats.ndim == 3
        assert pertmats.ndim == 2
        assert delvecs.ndim == 3

        # create a result
        sampletimes = metadata["samplepoints"]
        model_attributes = {
            "winsize": metadata["winsize"],
            "stepsize": metadata["stepsize"],
            "samplerate": metadata["samplerate"],
        }
        contacts = Contacts(chlabels, require_matching=False)
        resultobj = Result(
            pertmats,
            sampletimes,
            contacts,
            metadata=metadata,
            model_attributes=model_attributes,
        )
        return resultobj

    def load_file(
        self, filepath: Union[str, os.PathLike], jsonfpath: Union[str, os.PathLike]
    ):
        if filepath.endswith(".npz"):
            res = self.read_npzjson(jsonfpath, filepath)
        elif filepath.endswith(".npy"):
            res = self.read_npyjson(jsonfpath, filepath)
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
        npzfpath: Union[str, os.PathLike] = None,
        return_struct: bool = False,
    ) -> object:
        """
        Reads a numpy stored as npz+json file combination.

        Parameters
        ----------
        jsonfpath :
        npzfpath :
        return_struct :

        Returns
        -------

        """
        filedir = os.path.dirname(jsonfpath)
        # load in json file
        metadata = loadjsonfile(jsonfpath)

        if npzfpath == None:
            npzfilename = metadata["resultfilename"]
            npzfpath = os.path.join(filedir, npzfilename)

        datastruct = np.load(npzfpath)

        if return_struct:
            return datastruct, metadata
        else:
            resultobj = self._wrap_result_in_obj(datastruct, metadata)
            return resultobj

    def read_npyjson(
        self,
        jsonfpath: Union[str, os.PathLike],
        npyfpath: Union[str, os.PathLike] = None,
        return_struct: bool = False,
    ):
        """
        Reads a numpy stored as npy+json file combination.

        Parameters
        ----------
        jsonfpath :
        npyfpath :
        return_struct :

        Returns
        -------

        """
        filedir = os.path.dirname(jsonfpath)
        # load in json file
        metadata = loadjsonfile(jsonfpath)

        if npyfpath == None:
            npyfilename = metadata["resultfilename"]
            npyfpath = os.path.join(filedir, npyfilename)

        arr = np.load(npyfpath)

        if return_struct:
            return arr, metadata
        else:
            chlabels = metadata["chanlabels"]

            assert arr.ndim == 2
            assert arr.shape[0] == len(chlabels)

            # create a result
            sampletimes = metadata["samplepoints"]
            model_attributes = {
                "winsize": metadata["winsize"],
                "stepsize": metadata["stepsize"],
                "samplerate": metadata["samplerate"],
            }
            contacts = Contacts(chlabels, require_matching=False)
            resultobj = Result(
                arr,
                sampletimes,
                contacts,
                metadata=metadata,
                model_attributes=model_attributes,
            )
            return resultobj
