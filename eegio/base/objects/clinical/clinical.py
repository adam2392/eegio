# -*- coding: utf-8 -*-
import os
from pprint import pprint

from eegio.base.objects.clinical.baseclinical import DataSheet


def patient_check(f):
    def wrapper(*args):
        patientdict = args[0].patientdict
        patid = args[1]
        if patid not in patientdict.keys():
            raise RuntimeError(
                "{} is not in our clinical timeseries?! "
                "In Master_clinical.py".format(patid)
            )
        return f(*args)

    return wrapper


class DatasetClinical(DataSheet):
    """
    Dataset clinical metadata class for a single instance of a dataset.

    """

    def __init__(
        self,
        patid: str = None,
        datasetid=None,
        centerid=None,
        fpath: os.PathLike = None,
    ):
        super(DatasetClinical, self).__init__(fpath=fpath)
        self.patid = patid
        self.datasetid = datasetid
        self.centerid = centerid

    def summary(self):
        summary_str = (
            f"{self.patid} - {self.datasetid} from center: {self.centerid}. Located "
            f"at {self.fpath}"
        )
        pprint(summary_str)
        return summary_str


class PatientClinical(DataSheet):
    def __init__(
        self,
        patid: str = None,
        datasetlist=[],
        centerid=None,
        fpath: os.PathLike = None,
    ):
        super(PatientClinical, self).__init__(fpath=fpath)
        self.patid = patid
        self.centerid = centerid
        self.datasetlist = datasetlist

        self._populate_attr_fromdf()

    def _populate_attr_fromdf(self):
        if "patient_id" in self.df.columns:
            self.patid = self.df.patient_id[0]

    def summary(self):
        summary_str = (
            f"{self.patid} with {len(self.datasetlist)} datasets from center: {self.centerid}. "
            f"Located at {self.fpath}."
        )
        pprint(summary_str)
        return summary_str
