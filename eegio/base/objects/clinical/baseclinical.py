# -*- coding: utf-8 -*-
import os
import warnings
from abc import ABC, abstractmethod
from ast import literal_eval
from pprint import pprint
from sys import getsizeof

import pandas as pd


def format_list_str_channels(chanlist):
    return literal_eval(chanlist[0])


numerical_attributes = [
    'onset_age',
    'age_surgery',
]

MB = 1e6
GB = 1e9


class AbstractClinical(ABC):
    """
    Patient clinical metadata class for a single instance of a patient.
    """

    def __init__(self, patid, **kwargs):
        self.id = patid
        self._set_attr_from_kwargs(**kwargs)

    def __str__(self):
        self.summary()

    def _set_attr_from_kwargs(self, **kwargs):
        """
        Helper function to dynamically set attributes of the Patient Clinical class from
        keyword arguments.

        :param kwargs: (dict) dictionary of keyword arguments that get attached to this object
        :return: PatientClinical class object
        """
        # do some checking of the dictionary
        if getsizeof(kwargs) > 50 * MB:
            warnings.warn("Size of passed in dictionary is larger then 50MB. Possibly not a good idea."
                          "You passed in key values: {keys}".format(keys=kwargs.keys()))

        for (field, value) in kwargs.items():
            setattr(self, field, value)
        return self

    def load_from_dict(self, metadata_dict: dict):
        # call dynamic function to assign properties
        self._set_attr_from_kwargs(**metadata_dict)

    @abstractmethod
    def summary(self):
        raise NotImplementedError(
            "Needs to have a summary function that pretty prints.")

    def load_from_df(self, df: pd.DataFrame):
        # get dictionary from dataframe
        df_dict = df.to_dict(orient="dict",
                             instance="dict")

        # call dynamic function to assign properties
        self._set_attr_from_kwargs(df_dict)

    def load_from_excel(self, filepath: os.PathLike, **read_excel_kwargs):
        # load in excel file
        df = pd.read_excel(filepath, **read_excel_kwargs)

        # convert to dictionary
        self.load_from_df(df)


class DatasetClinical(AbstractClinical):
    """
    Dataset clinical metadata class for a single instance of a dataset.

    """

    def __init__(self, patid, datasetid=None, centerid=None):
        super(DatasetClinical, self).__init__(patid)

        self.datasetid = datasetid
        self.centerid = centerid


class PatientClinical(AbstractClinical):
    def __init__(self, patid, datasetlist=[], centerid=None):
        super(PatientClinical, self).__init__(patid)

        self.centerid = centerid
        self.datasetlist = datasetlist

    def summary(self):
        summary_str = f"{self.id} with {len(self.datasetlist)} datasets from center: {self.centerid}. " \
                      f""
        pprint(summary_str)


if __name__ == '__main__':
    import numpy as np

    filepath = "../../../tests/organized_datasheet_formatted.xlsx"

    patid = "pat01"
    datasetid = "sz_2"

    datasetids = ['sz_2', 'sz_3']
    centerid = "jhu"
    example_datadict = {
        'length_of_recording': 400,
        'timepoints': np.hstack((np.arange(0, 100), np.arange(5, 105))),
    }

    patclin = PatientClinical(patid, datasetids, centerid)
    patclin.load_from_dict(example_datadict)

    for key, val in example_datadict.items():
        testval = getattr(patclin, key)
        # assert testval == val

    print(dir(patclin))
