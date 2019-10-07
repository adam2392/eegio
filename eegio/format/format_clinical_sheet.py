import os
import re
from typing import Union, List

import numpy as np
import pandas as pd


def expand_channels(ch_list):
    ch_list = [a.replace("’", "'").replace("\n", "").replace(" ", "") for a in ch_list]

    new_list = []
    for string in ch_list:
        if string == "nan":
            continue

        if not string.strip():
            continue

        # A'1,2,5,7
        match = re.match("^([A-Za-z]+[']*)([0-9,]*)([A-Za-z]*)$", string)
        if match:
            name, fst_idx, last_idx = match.groups()
            numbers = fst_idx.split(",")
            new_list.extend([name + str(char) for char in numbers if char != ","])
            continue

        # A'1
        match = re.match("^([A-Za-z]+[']*)([0-9]+)$", string)
        if match:
            new_list.append(string)
            continue

        # A'1-10
        match = re.match("^([A-Za-z]+[']*)([0-9]+)-([0-9]+)$", string)
        if match:
            name, fst_idx, last_idx = match.groups()
            new_list.extend(
                [name + str(i) for i in range(int(fst_idx), int(last_idx) + 1)]
            )
            continue

        # A'1-A10
        match = re.match("^([A-Za-z]+[']*)([0-9]+)-([A-Za-z]+[']*)([0-9]+)$", string)
        if match:
            name1, fst_idx, name2, last_idx = match.groups()
            if name1 == name2:
                new_list.extend(
                    [name1 + str(i) for i in range(int(fst_idx), int(last_idx) + 1)]
                )
                continue

        # A'1,B'1,
        match = re.match("^([A-Za-z]+[']*)([0-9,])([A-Za-z]*)$", string)
        if match:
            name, fst_idx, last_idx = match.groups()
            numbers = fst_idx.split(",")
            new_list.extend([name + str(char) for char in numbers if char != ","])
            continue

        match = string.split(",")
        if match:
            new_list.extend([ch for ch in match])
            continue
        print("expand_channels: Cannot parse this: %s" % string)
    return new_list


def format_clinical_sheet(
    fpath: Union[str, os.PathLike],
    cols_to_reg_expand: List = None,
    patientid: str = None,
    # return_bad_contacts: bool = True,
):
    formatter = FormatClinicalSheet(fpath, cols_to_reg_expand, patientid)
    return formatter.df.to_dict()


class FormatClinicalSheet:
    def __init__(
        self,
        fpath: Union[str, os.PathLike],
        cols_to_reg_expand: List = None,
        patientid: str = None,
    ):
        self.fpath = fpath
        self.patientid = patientid
        self.cols_to_expand = cols_to_reg_expand

        self.read_file(self.fpath)

    def read_file(self, fpath):
        if fpath.endswith(".csv"):
            df = pd.read_csv(fpath, header=0, index_col=False)
        elif fpath.endswith(".xlsx"):
            df = pd.read_excel(fpath, header=0, index_col=False)
        else:
            raise RuntimeError(
                f"Can't read in files not with ending: csv, or xlsx."
                f" You passed file: {fpath}."
            )
        self.df = df

        # format column headers
        self._format_col_headers()

        # expand channel annotations
        self._expand_ch_annotations()

        # if patient id passed in, only get row w/ that patient id
        if self.patientid != None:
            self.df = self.df[self.df["patient_id"] == self.patientid]

    def _format_col_headers(self):
        self.df = self.df.apply(lambda x: x.astype(str).str.lower())
        # map all empty to nans
        self.df = self.df.fillna(np.nan)
        self.df = self.df.replace("nan", "", regex=True)

    def _expand_ch_annotations(self):
        if self.cols_to_expand == None:
            return

        # do some string processing to expand out contacts
        for col in self.cols_to_expand:
            # strip out blank spacing
            self.df[col] = self.df[col].str.strip()
            # split contacts by ";", ":", or ","
            self.df[col] = self.df[col].str.split("; |: |,")
            self.df[col] = self.df[col].map(lambda x: [y.strip() for y in x])
            self.df[col] = self.df[col].map(lambda x: [y.replace(" ", "-") for y in x])

            # expand channel labels
            self.df[col] = self.df[col].apply(lambda x: expand_channels(x))
