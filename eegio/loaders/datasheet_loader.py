# -*- coding: utf-8 -*-
import os
import warnings
from abc import ABC, abstractmethod
from ast import literal_eval
from pprint import pprint
from sys import getsizeof

import pandas as pd

from eegio.base.config import MB


def format_list_str_channels(chanlist):
    return literal_eval(chanlist[0])


numerical_attributes = ["onset_age", "age_surgery"]


class AbstractClinical(ABC):
    """
    Patient clinical metadata class for a single instance of a patient.
    """

    def __init__(self, **kwargs):
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
            warnings.warn(
                "Size of passed in dictionary is larger then 50MB. Possibly not a good idea."
                "You passed in key values: {keys}".format(keys=kwargs.keys())
            )

        for (field, value) in kwargs.items():
            setattr(self, field, value)
        return self

    def load_from_dict(self, metadata_dict: dict):
        # call dynamic function to assign properties
        self._set_attr_from_kwargs(**metadata_dict)

    @abstractmethod
    def summary(self):
        raise NotImplementedError(
            "Needs to have a summary function that pretty prints."
        )

    def load_from_df(self, df: pd.DataFrame):
        # get dictionary from dataframe
        df_dict = df.to_dict(orient="dict", instance="dict")

        # call dynamic function to assign properties
        self._set_attr_from_kwargs(df_dict)

    def from_excel(self, filepath: os.PathLike, cache_vars=False, **read_excel_kwargs):
        # load in excel file
        df = pd.read_excel(filepath, **read_excel_kwargs)

        # convert to dictionary
        if cache_vars:
            self.load_from_df(df)
        return df

    def from_csv(
        self, filepath: os.PathLike, cache_vars: bool = False, **read_csv_kwargs
    ):
        # load in csv file
        df = pd.read_csv(filepath, **read_csv_kwargs)

        if cache_vars:
            self.load_from_df(df)
        return df

    def _filter_column_name(self, name):
        # strip parentheses
        name = name.split("(")[0]

        # strip whitespace
        name = name.strip()

        return name

    def clean_columns(self, df):
        # lower-case column names
        df.rename(str.lower, axis="columns", inplace=True)

        column_names = df.columns
        column_mapper = {name: self._filter_column_name(name) for name in column_names}
        # remove any markers (extra on clinical excel sheet)
        df.rename(columns=column_mapper, errors="raise", inplace=True)
        return df


class DataSheet(AbstractClinical):
    def __init__(self, fpath: os.PathLike = None):
        if fpath != None:
            super(DataSheet, self).__init__(fpath=fpath)
            self.load(fpath)

    def load(self, fpath: os.PathLike):
        fext = os.path.splitext(fpath)[1]
        if fext == ".csv":
            df = self.from_csv(fpath)
        elif fext == ".xlsx":
            df = self.from_excel(fpath)
        elif fext == ".txt":
            df = self.from_csv(fpath)
        else:
            raise AttributeError(
                f"No loading functionality set for {fext} files. "
                f"Currently supports: csv, xlsx, txt."
            )
        self.df = df
        return df

    def summary(self):
        summary_str = f"Datasheet located at {self.fpath}."
        pprint(summary_str)
        return summary_str
