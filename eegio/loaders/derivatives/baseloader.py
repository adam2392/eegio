import os
from abc import ABC, abstractmethod


class BaseLoader(ABC):
    """
    Class for a base data converter.

    It mainly handles universal text parsing and
    preprocessing of files done to get to our final desired file format (i.e. fif+json file pair)
    per timeseries.
    """

    def __init__(self, fname: os.PathLike = None):
        self.fname = fname
        self.metadata_dict = dict()

    def _set_metadatadict(self):
        self.metadata_dict.update(fname=self.fname)

    def get_metadata(self):
        """Get metadata dictionary."""
        return self.metadata_dict

    def get_size(self):
        """Get size of the dataset to be loaded."""
        MB = 1.0e6
        GB = 1e9
        if self.fname is not None:
            fsize = os.path.getsize(self.fname)
            return fsize / MB
        return None

    # @abstractmethod
    def read_NK(self, fname: os.PathLike):
        """Convert from Nihon Kohden systems."""
        pass

    # @abstractmethod
    def read_Natus(self, fname: os.PathLike):
        """Convert from Natus systems."""
        pass

    @abstractmethod
    def load_file(self, filepath: os.PathLike):
        """
        Abstract method for loading a file.

        Needs to be implemented by any data converters that will load a file to convert.

        Parameters
        ----------
        filepath : os.PathLike
            filepath for the dataset to load.

        """
        raise NotImplementedError(
            "Implement function for loading in file for starting conversion!"
        )

    def update_metadata(self, **update_kws):
        """
        Keyword argument update function for the metadata_dict property.

        Parameters
        ----------
        update_kws : Dict
            keyword arguments to update metadata_dict with.

        """
        self.metadata_dict.update(**update_kws)
