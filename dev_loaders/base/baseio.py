# -*- coding: utf-8 -*-
import io
import json

import numpy as np
import scipy.io as sio
from eegio.base.utils.data_structures_utils import NumpyEncoder


class BaseIO(object):
    """
    Base Input/Output class that wraps functions for mainly saving arraydata + metadata easily as:

    - numpy structs
    - json files
    - mat files
    - hdf files

    """

    def _load_pickled_object(self, fpath):
        pass

    def _writejsonfile(self, metadata, metafilename):
        """
        Helper method for writing to a json file with a dictionary metadata.

        :param metadata: (dict) metadata dictionary
        :param metafilename: (str) outputfilepath to write data to
        :return: None
        """
        with io.open(metafilename, "w", encoding="utf8") as outfile:
            str_ = json.dumps(
                metadata,
                indent=4,
                sort_keys=True,
                cls=NumpyEncoder,
                separators=(",", ": "),
                ensure_ascii=False,
            )
            outfile.write(str_)

    def _loadjsonfile(self, metafilename):
        """
        Helper method for loading json file with metadata written.

        Loads it into a dictionary.

        :param metafilename: (str) filepath to the metadata
        :return: metadata (dict) a dictionary-style metadata data structure
        """
        if not metafilename.endswith(".json"):
            metafilename += ".json"

        try:
            with open(metafilename, mode="r", encoding="utf8") as f:
                metadata = json.load(f)
            metadata = json.loads(metadata)
        except:
            with io.open(metafilename, encoding="utf-8", mode="r") as fp:
                json_str = fp.read()  # json.loads(
            metadata = json.loads(json_str)

        return metadata

    def _loadnpzfile(self, npzfilename):
        """
        Helper method to load a numpy zipped file.

        :param npzfilename: (str) filepath
        :return: (dict) a dictionary for the numpy arrays
        """
        if npzfilename.endswith(".json"):
            npzfilename = npzfilename.split(".json")[0]
        if not npzfilename.endswith(".npz"):
            npzfilename += ".npz"
        result_struct = dict(np.load(npzfilename, encoding="latin1"))
        return result_struct

    def _writenpzfile(self, npzfilename, kwargs):
        """
        Helper method to write a numpy zipped file.

        :param npzfilename:
        :param kwargs:
        :return:
        """
        if not npzfilename.endswith(".npz"):
            npzfilename += ".npz"
        np.savez_compressed(npzfilename, **kwargs)

    def _writematfile(self, matfilename, kwargs):
        """
        Helper function used to write to a mat file.

        We will need the matrix CxT and contact regs of Cx1
        and also the regmap vector that will be Nx1.

        or

        matrix of RxT (averaged based on contact_regs) and the Nx1
        reg map vector that will be used to map each of the vertices
        to a region in R.

        Using these, matlab can easily assign the triangles that
        belong to regions for each channel/region and color with
        that color according to the colormap defined by the function.

        :param matfilename: (str) filepath to write matlab struct to.
        :param kwargs:
        :return: None
        """
        if not matfilename.endswith(".mat"):
            matfilename += ".mat"
        sio.savemat(matfilename, squeeze_me=True, struct_as_record=True, **kwargs)
