import os
from abc import ABC, abstractmethod


class BaseLoader(ABC):
    """
    Class for a base data converter. It mainly handles universal text parsing and
    preprocessing of files done to get to our final desired file format (i.e. fif+json file pair)
    per timeseries.
    """

    def __init__(self, fname: os.PathLike = None):
        self.fname = fname
        self.metadata = None

    def get_size(self):
        MB = 1.0e6
        GB = 1e9
        if self.fname is not None:
            fsize = os.path.getsize(self.fname)
            return fsize / MB
        return None

    @abstractmethod
    def read_NK(self, fname: os.PathLike):
        pass

    @abstractmethod
    def read_Natus(self, fname: os.PathLike):
        pass

    @abstractmethod
    def load_file(self, filepath: os.PathLike):
        """
        Abstract method for loading a file. Needs to be implemented by any data converters that will
        load a file to convert.

        :return: None
        """
        raise NotImplementedError(
            "Implement function for loading in file for starting conversion!")

    def updatemetadata(self, update_kws: dict = {}):
        if self.metadata != None:
            for key in update_kws.keys():
                self.metadata[key] = update_kws[key]
