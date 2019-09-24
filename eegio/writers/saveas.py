import datetime
import os
from typing import List, Dict

import mne
import numpy as np
import pyedflib

from eegio.base.config import DATE_MODIFIED_KEY
from eegio.writers.basewrite import BaseWrite

def _check_hd5py():
    try:
        import h5py as hpy
    except ImportError as e:
        raise ImportError("Need to download h5py if you want to use this.")
    return hpy

def get_tempfilename(x, ext):
    return f"temp_{x}.{ext}"


class TempWriter(BaseWrite):
    def __init__(self, tempdir: os.PathLike = None):
        self.tempdir = tempdir

    @classmethod
    def save_npz_file(cls, fdir: os.PathLike, index: int, compress=False, **kwds):
        if index == None:
            raise RuntimeError(
                "Need to pass in a filepath to save, or an index "
                "of the file to save. E.g. If you want to save temporary"
                " arrays in sequence, pass in a sequentially increasing index."
            )
        tempfilename = os.path.join(fdir, get_tempfilename(index, ext="npz"))

        if compress:
            np.savez_compressed(tempfilename, **kwds)
        else:
            np.savez(tempfilename, **kwds)
        return tempfilename

    @classmethod
    def save_npy_file(cls, fdir: os.PathLike, index: int, arr: np.ndarray):
        """
        Temporary writer to a .npy binary file. This provides fast loading/saving of the arrays, since
        we don't need to save multiple keyword arguments to a .npz file. Fix_imports is set as False, so
        there is no compatability with Python2.

        :param fdir:
        :type fdir:
        :param index:
        :type index:
        :param arr:
        :type arr:
        :return:
        :rtype:
        """
        if index == None:
            raise RuntimeError(
                "Need to pass in a filepath to save, or an index "
                "of the file to save. E.g. If you want to save temporary"
                " arrays in sequence, pass in a sequentially increasing index."
            )
        tempfilename = os.path.join(fdir, get_tempfilename(index, ext="npy"))
        np.save(tempfilename, arr, fix_imports=False)
        return tempfilename


class DataWriter(BaseWrite):
    def __init__(self, fpath=None, raw: mne.io.BaseRaw = None, type: str = "fif"):
        if raw != None and fpath == None:
            raise RuntimeError("Pass in a file path to save data!")

        if fpath != None and raw == None:
            raise RuntimeError("Pass in a MNE Raw object to save!")

        if fpath != None and raw != None:
            if not os.path.exists(os.path.dirname(fpath)):
                fdir = os.path.dirname(fpath)
                raise RuntimeError(
                    "Filepath you passed to save data does not exist. Please "
                    f"first create the corresponding directory: {fdir}"
                )
            if type == "fif":
                self.saveas_fif(fpath, raw.get_data(return_times=False), raw.info)

    def saveas_hd5(self, fpath, rawdata, info, group=None):
        h5py = _check_hd5py()

        with h5py.File("mytestfile.hdf5", "w") as f:
            if group != None:
                dset = f.create_dataset("mydataset", (100,), dtype='i')
        pass

    def saveas_fif(self, fpath, rawdata, info, bad_chans_list=[], montage: List = []):
        """
        Conversion function for the rawdata + metadata into a .fif file format. The accompanying metadata .json
        file will be handled in the convert_metadata() function.

        rawdata.edf -> .fif

        :param bad_chans_list: (optional; list) a list of the bad channels string
        :param fpath: (optional; os.PathLike) the file path for the converted fif data
        :param save: (optional; bool) to save the output fif dataset or not

        :return: formatted_raw (mne.Raw) the raw fif dataset
        """
        # perform check on the info data struct
        self._check_info(info)

        # save the actual raw array
        formatted_raw = mne.io.RawArray(rawdata, info, verbose="ERROR")

        fmt = "single"
        formatted_raw.save(fpath, overwrite=True, fmt=fmt, verbose="ERROR")
        return formatted_raw

    def saveas_edf(self, fpath, rawdata, info, bad_chans_list=[], montage: List = []):
        pass

    def _pyedf_saveas_edf(
        mne_raw, fname, events_list, picks=None, tmin=0, tmax=None, overwrite=False
    ):
        """
        Saves the raw content of an MNE.io.Raw and its subclasses to
        a file using the EDF+ filetype
        pyEDFlib is used to save the raw contents of the RawArray to disk
        Parameters
        ----------
        mne_raw : mne.io.Raw
            An object with super class mne.io.Raw that contains the data
            to save
        fname : string
            File name of the new dataset. This has to be a new filename
            unless data have been preloaded. Filenames should end with .edf
        picks : array-like of int | None
            Indices of channels to include. If None all channels are kept.
        tmin : float | None
            Time in seconds of first sample to save. If None first sample
            is used.
        tmax : float | None
            Time in seconds of last sample to save. If None last sample
            is used.
        overwrite : bool
            If True, the destination file (if it exists) will be overwritten.
            If False (default), an error will be raised if the file exists.
        """
        if not issubclass(type(mne_raw), mne.io.BaseRaw):
            raise TypeError("Must be mne.io.Raw type")
        if not overwrite and os.path.exists(fname):
            raise OSError("File already exists. No overwrite.")
        # static settings
        file_type = pyedflib.FILETYPE_EDFPLUS
        sfreq = mne_raw.info["sfreq"]
        date = datetime.now().strftime("%d %b %Y %H:%M:%S")
        first_sample = int(sfreq * tmin)
        last_sample = int(sfreq * tmax) if tmax is not None else None

        # convert data
        channels = mne_raw.get_data(picks, start=first_sample, stop=last_sample)

        # convert to microvolts to scale up precision
        channels *= 1e6

        # set conversion parameters
        dmin, dmax = [-32768, 32767]
        pmin, pmax = [channels.min(), channels.max()]
        n_channels = len(channels)

        # create channel from this
        try:
            f = pyedflib.EdfWriter(fname, n_channels=n_channels, file_type=file_type)

            channel_info = []
            data_list = []

            for i in range(n_channels):
                ch_dict = {
                    "label": mne_raw.ch_names[i],
                    "dimension": "uV",
                    "sample_rate": sfreq,
                    "physical_min": pmin,
                    "physical_max": pmax,
                    "digital_min": dmin,
                    "digital_max": dmax,
                    "transducer": "",
                    "prefilter": "",
                }
                #             print(ch_dict)
                channel_info.append(ch_dict)
                data_list.append(channels[i])

            f.setTechnician("mne_save_edf_adamli")
            f.setSignalHeaders(channel_info)
            for event in events_list:
                onset_in_seconds, duration_in_seconds, description = event
                print(onset_in_seconds, duration_in_seconds, description)
                f.writeAnnotation(
                    float(onset_in_seconds), int(duration_in_seconds), description
                )
            f.setStartdatetime(date)
            f.writeSamples(data_list)
        except Exception as e:
            print(e)
            return False
        finally:
            f.close()
        return True

    def merge_npy_arrays(
        self, outputfpath: str, fpathlist: List, metadata: Dict, resname: str = "result"
    ):
        def check_equal(arr1, arr2):
            pass

        # store the date that this dataset was computed
        metadata[DATE_MODIFIED_KEY] = datetime.datetime.now()

        merged_arr = []
        for i, fpath in enumerate(fpathlist):
            arr = np.load(fpath)
            if i > 0:
                # do some checking
                check_equal(arr, merged_arr[-1])

            merged_arr.append(arr)

        # save
        kwd_arrs = {resname: merged_arr, "metadata": metadata}

        np.savez_compressed(outputfpath, **kwd_arrs)

        return merged_arr
