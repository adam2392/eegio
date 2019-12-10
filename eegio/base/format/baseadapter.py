import os
from abc import ABC

from eegio.base.config import MB


class BaseAdapter(ABC):
    """
    Class for a base data adapter.

    It handles the conversion of the data into a python readable format:
    - timeseries -> numpy.ndarray
    - metadata -> collection of dictionaries

    """

    def __init__(self, type: str):
        self.type = type

    def get_size(self, fname):
        """
        Get total size of datasets in MB.

        Parameters
        ----------
        fname :

        Returns
        -------
        fsize : the file size in MB

        """
        if os.path.exists(fname):
            fsize = os.path.getsize(fname)
            return fsize / MB
        else:
            raise ValueError("fname should be a valid existing path!")
