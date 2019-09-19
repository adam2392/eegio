import os
from abc import ABC, abstractmethod
from eegio.base.config import MB, GB


class BaseAdapter(ABC):
    """
    Class for a base data adapter. It handles the conversion of the data into a python readable format:

    - timeseries -> numpy.ndarray
    - metadata -> collection of dictionaries
    """

    def __init__(self, type: str):
        self.type = type

    def get_size(self, fname):
        if os.path.exists(fname):
            fsize = os.path.getsize(fname)
            return fsize / MB
        else:
            raise ValueError("fname should be a valid existing path!")

    @abstractmethod
    def convert_tsdata(self, filepath: os.PathLike):
        raise NotImplementedError(
            "Implement function for loading in file for starting conversion!"
        )

    @abstractmethod
    def convert_metadata(self, filepath: os.PathLike):
        raise NotImplementedError(
            "Implement function for loading in file for starting conversion!"
        )
