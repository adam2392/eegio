from pprint import pprint

import pandas as pd


class DatasetClinical:
    """
    Dataset clinical metadata class for a single instance of a dataset.

    """

    def __init__(
        self, patid: str = None, datasetid=None, centerid=None, df: pd.DataFrame = None
    ):
        self.patid = patid
        self.datasetid = datasetid
        self.centerid = centerid
        self.df = df

    def summary(self):
        summary_str = (
            f"{self.patid} - {self.datasetid} from center: {self.centerid}. Located "
        )
        pprint(summary_str)
        return summary_str


class PatientClinical:
    def __init__(
        self, patid: str = None, datasetlist=[], centerid=None, df: pd.DataFrame = None
    ):
        self.patid = patid
        self.centerid = centerid
        self.datasetlist = datasetlist
        self.df = df

        self._populate_attr_fromdf()

    def _populate_attr_fromdf(self):
        if "patient_id" in self.df.columns:
            self.patid = self.df.patient_id[0]

    def summary(self):
        summary_str = f"{self.patid} with {len(self.datasetlist)} datasets from center: {self.centerid}. "
        pprint(summary_str)
        return summary_str
