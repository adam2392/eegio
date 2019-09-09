# import os
# import tempfile
#
# import pytest
# import numpy as np
#
# from base.config.dataconfig import IEEG_REFERENCES
# from eegio.format.dataset_tester import DatasetTester
# from eegio.format.formatter_raw import ConvertEDFiEEG
# from eegio.dev.dataset.timeseries.ieegrecording import iEEGRecording
#
# centers = [
#     'nih',
#     'jhu',
#     'ummc',
#     'cleveland',
#     'clevelandnl',
# ]
# modalities = [
#     'seeg',
#     'scalp',
# ]
# RAWDATADIR = "/Users/adam2392/Downloads/tngpipeline"
#
#
# @pytest.mark.usefixture('edffilepath')
# class Test_RawComputation_Endpoints():
#     """
#     Testing class for testing all the raw data that was generated through preprocessing.
#     - fif + json datasets
#
#     Runs the eegio.format.dataset_tester
#
#     TODO:
#     1. Figure out why raw_fif and the original data is slightly different numerically.
#     Saving of fif dataset?
#
#     """
#
#     def test_rawdata_fromedf(self, edffilepath):
#         pat_id = 'pt1'
#         dataset_id = 'sz2'
#         clinical_center = 'nih'
#         datatype = 'ieeg'
#
#         # create temporary directory to save data
#         newfifname = f"{pat_id}_{dataset_id}_raw.fif"
#
#         # initialize converter
#         edfconverter = ConvertEDFiEEG(datatype=datatype)
#         # load in the dataset and create metadata object
#         edfconverter.load_file(filepath=edffilepath)
#         edfconverter.extract_info_and_events()
#
#         # create temporary directory
#         with tempfile.TemporaryDirectory(dir="./") as dirpath:
#             newfifpath = os.path.join(dirpath, newfifname)
#             newjsonpath = os.path.join(
#                 dirpath, newfifname.replace('_raw.fif', '.json'))
#
#             # test saving the data
#             rawfif = edfconverter.convert_fif(
#                 fpath=newfifpath, save=True, replace=True)
#             metadata = edfconverter.convert_metadata(
#                 pat_id, dataset_id, clinical_center, save=True, jsonfilepath=newjsonpath)
#
#             # test loading in the actual data from the fif saved place
#             loader = iEEGRecording(root_dir=dirpath,
#                                    jsonfilepath=newjsonpath,
#                                    preload=False,
#                                    reference=IEEG_REFERENCES.monopolar.value)
#             ieegts = loader.loadpipeline()
#
#             # assert data is save
#             Adata = ieegts.get_data()
#             Bdata = rawfif.get_data()[loader.rawmask_inds, :]
#
#             # run filtering to make sure they undergo same data transformations
#             Bdata = loader.filter_data(
#                 Bdata, ieegts.samplerate, ieegts.linefreq)
#
#         print(Adata.shape, Bdata.shape)
#         print(Adata.dtype, Bdata.dtype)
#         print(np.max(Adata), np.min(Adata), np.max(Bdata), np.min(Bdata))
#         assert Adata.shape == Bdata.shape
#         # assert np.linalg.norm((Adata - Bdata)[:], ord=np.inf) < 1e-6
#
#     def test_rawdata(self):
#         dataset_tester = DatasetTester(modality='ieeg')
#
#         all_test_results = []
#
#         # test each center
#         for center in centers:
#             centerdir = os.path.join(RAWDATADIR, center)
#             centerpats = [f for f in os.listdir(centerdir) if os.path.isdir(f)]
#
#             for pat_id in centerpats:
#                 patdir = os.path.join(centerdir, pat_id)
#
#                 for modality in modalities:
#                     datadir = os.path.join(patdir, modality, "fif")
#                     patdatasets_jsonfiles = [f for f in os.listdir(
#                         datadir) if f.endswith('.json')]
#
#                     for jsonpath in patdatasets_jsonfiles:
#                         # test loading in the actual data from the fif saved place
#                         loader = iEEGRecording(root_dir=datadir,
#                                                jsonfilepath=jsonpath,
#                                                preload=False,
#                                                reference=IEEG_REFERENCES.monopolar.value)
#                         ieegts = loader.loadpipeline()
#
#                         # assert data is save
#                         rawdata = ieegts.get_data()
#                         metadata = ieegts.get_metadata()
#
#                         # test the metadata for this dataset
#                         dataset_tester.load_datadict(metadata)
#                         dataset_tester.test_pipeline()
#                         test_results = dataset_tester.get_test_results()
#
#                         if test_results != []:
#                             all_test_results.append(test_results)
#
#         print("The results of testing all the fif/json datasets in all centers: \n", all_test_results)
#         assert all_test_results == []
