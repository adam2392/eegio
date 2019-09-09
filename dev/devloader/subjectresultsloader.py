import os
import warnings

from eegio.loaders.clinical import ExcelReader
from eegio.loaders.dataset.resultloader import ResultLoader
from eegio.loaders.patient.subjectbase import SubjectBase


class SubjectResultsLoader(SubjectBase):
    """
    Patient level wrapper for loading in data, or result. It helps
    load in all the datasets for a particular subject, for doing per-subject analysis.

    Assumes directory structure:
    root_dir
        -> name
            -> datasets / result_computes
    """

    def __init__(self, root_dir,
                 subjid=None,
                 datatype='frag',
                 preload=True,
                 use_excel_meta=False):
        # if datatype not in MODELDATATYPES:
        #     raise ValueError("Type of subject loader must be one of {}. You passed in {}!".format(MODELDATATYPES, type))

        super(SubjectResultsLoader, self).__init__(
            root_dir=root_dir, subjid=subjid, datatype=datatype)

        self.use_excel_meta = use_excel_meta

        # set reference for this recording and jsonfilepath
        self.loadingfunc = ResultLoader
        self.datatype = datatype

        # get a list of all the jsonfilepaths possible
        self._getalljsonfilepaths(self.root_dir, subjid=subjid)

        print("Found {} jsonfilepaths for {}".format(
            len(self.jsonfilepaths), self.subjid))
        if preload:
            self.read_all_files()

    def read_all_files(self):
        if not self.is_loaded:
            # load in each file / timeseries separately
            for idx, json_file in enumerate(self.jsonfilepaths):
                jsonfilepath = os.path.join(self.root_dir, json_file)

                if not os.path.exists(self._get_corresponding_npzfile(jsonfilepath)):
                    print("Data file for {} does not exist!".format(jsonfilepath))
                    continue

                result = self.loadingfunc(results_dir=self.root_dir,
                                          jsonfilepath=jsonfilepath,
                                          datatype=self.datatype,
                                          preload=False)
                resultmodel = result.loadpipeline(jsonfilepath=jsonfilepath)

                if resultmodel.onsetwin == resultmodel.offsetwin:
                    warnings.warn(
                        "Result model onsetwin == offsetwin in {}".format(jsonfilepath))
                    continue

                if self.use_excel_meta:
                    print(
                        "TODO: REFORMAT THIS FUNCTION. Bringing in excel data for onsetchans")
                    resultmodel.clinonsetlabels = self.load_onsetchans_fromexcel(
                        resultmodel.patient_id)

                # create data structures of the results
                self.datasets.append(resultmodel)
                self.dataset_ids.append(resultmodel.dataset_id)
        else:
            raise RuntimeError(
                "Datasets for this patient are already loaded! Need to run reset() first!")

    def trim_datasets(self, resultmodel, offset_sec):
        resultmodel.trim_aroundonset(offset_sec=offset_sec)

        data = resultmodel.get_data()
        metadata = resultmodel.get_data()

        return data, metadata

    def load_onsetchans_fromexcel(self, patid):
        datafile = os.path.join(
            "/Users/adam2392/Dropbox/phd research/Fragility Analysis Project/datasheet_manual_notes.xlsx")
        dropboxdir = "/Users/adam2392/Dropbox/phd_research/Fragility_Analysis_Project/"
        alldatafile = os.path.join(
            dropboxdir, "organized_clinical_datasheet_formatted.xlsx")

        from ast import literal_eval
        sheetreader = ExcelReader(alldatafile)
        df = sheetreader.df
        ezcontacts = df[df['patid'] == patid]['ez_hypo_contacts'].values[0]
        ezcontacts = literal_eval(ezcontacts)

        return ezcontacts
