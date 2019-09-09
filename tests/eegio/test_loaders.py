# import pytest
#
# from base.config.dataconfig import IEEG_REFERENCES, SCALP_REFERENCES
# from eegio.dev.dataset.result.resultloader import ResultLoader
# from eegio.dev.dataset.timeseries.ieegrecording import iEEGRecording
# from eegio.dev.dataset.timeseries.scalprecording import ScalpRecording
# from eegio.dev.patient.subjectrawloader import SubjectRawLoader
#
#
# # @pytest.mark.skip(reason="Already tested! 12/14/18.")
# @pytest.mark.usefixture('ieeg_data_fif', 'scalp_data_fif', 'seeg_data_fif')
# class TestRawLoader():
#     def test_ieegrecordingloader(self, ieeg_data_fif):
#         testdatadir, testfilename = ieeg_data_fif
#         loader = iEEGRecording(root_dir=testdatadir,
#                                jsonfilepath=testfilename,
#                                preload=False,
#                                reference=IEEG_REFERENCES.monopolar.value)
#         assert loader.is_loaded is False
#         ts = loader.loadpipeline()
#         assert ts.reference == IEEG_REFERENCES.monopolar.value
#         assert ts.get_data().shape[0] == ts.ncontacts
#
#         # test bipolar reference
#         loader = iEEGRecording(root_dir=testdatadir,
#                                jsonfilepath=testfilename,
#                                preload=False,
#                                reference=IEEG_REFERENCES.bipolar.value)
#         assert loader.is_loaded is False
#         ts = loader.loadpipeline()
#         assert ts.reference == IEEG_REFERENCES.bipolar.value
#         assert ts.get_data().shape[0] == ts.ncontacts
#
#         # test common_avg refrence
#         loader = iEEGRecording(root_dir=testdatadir,
#                                jsonfilepath=testfilename,
#                                preload=False,
#                                reference=IEEG_REFERENCES.common_avg.value)
#         assert loader.is_loaded is False
#         ts = loader.loadpipeline()
#         assert ts.reference == IEEG_REFERENCES.common_avg.value
#         assert ts.get_data().shape[0] == ts.ncontacts
#
#     def test_scalprecordingloader(self, scalp_data_fif):
#         testdatadir, testfilename = scalp_data_fif
#         loader = ScalpRecording(root_dir=testdatadir,
#                                 jsonfilepath=testfilename,
#                                 preload=False,
#                                 reference=SCALP_REFERENCES.monopolar.value)
#         assert loader.is_loaded is False
#         ts = loader.loadpipeline()
#         assert ts.reference == SCALP_REFERENCES.monopolar.value
#         assert ts.get_data().shape[0] == ts.ncontacts
#
#         # test all references
#         loader = ScalpRecording(root_dir=testdatadir,
#                                 jsonfilepath=testfilename,
#                                 preload=False,
#                                 reference=SCALP_REFERENCES.common_avg.value)
#         assert loader.is_loaded is False
#         ts = loader.loadpipeline()
#         assert ts.reference == SCALP_REFERENCES.common_avg.value
#         assert ts.get_data().shape[0] == ts.ncontacts
#
#     def test_ieeg_channel_transformations(self, seeg_data_fif):
#         testdatadir, testfilename = seeg_data_fif
#         loader = iEEGRecording(root_dir=testdatadir,
#                                jsonfilepath=testfilename,
#                                preload=False, remove_wm_contacts=True,
#                                reference=IEEG_REFERENCES.monopolar.value)
#         ts = loader.loadpipeline()
#         assert ts.get_data().shape[0] == ts.ncontacts
#         assert len(ts.metadata['chanlabels']) == ts.ncontacts
#
#
# # @pytest.mark.skip(reason="Already tested! 12/14/18.")
# @pytest.mark.usefixtures('result_frag_files')
# class TestResultLoader():
#     def test_frag_resultloader(self, result_frag_files):
#         testdatadir, testfilename = result_frag_files
#         loader = ResultLoader(results_dir=testdatadir,
#                               jsonfilepath=testfilename,
#                               preload=False,
#                               datatype='frag',
#                               storagetype='numpy')
#         assert loader.is_loaded is False
#         resultmodel = loader.loadpipeline()
#         assert resultmodel.get_data().shape[0] == resultmodel.ncontacts
#         assert resultmodel.get_data().shape[-1] == resultmodel.numwins
#
#         assert resultmodel.cezcontacts or resultmodel.cezcontacts == []
#         assert isinstance(resultmodel.clinicaldifficulty, int)
#         assert isinstance(resultmodel.clinicalmatching, int)
#         assert isinstance(resultmodel.outcome, str)
#
#     def test_ltv_resultloader(self, result_frag_files):
#         testdatadir, testfilename = result_frag_files
#         loader = ResultLoader(results_dir=testdatadir,
#                               jsonfilepath=testfilename,
#                               preload=False,
#                               datatype='ltv',
#                               storagetype='numpy')
#         assert loader.is_loaded is False
#         resultmodel = loader.loadpipeline()
#         assert resultmodel.get_data().shape[0] == resultmodel.ncontacts
#         assert resultmodel.get_data().shape[-1] == resultmodel.numwins
#
#         assert resultmodel.cezcontacts or resultmodel.cezcontacts == []
#         assert isinstance(resultmodel.clinicaldifficulty, int)
#         assert isinstance(resultmodel.clinicalmatching, int)
#         assert isinstance(resultmodel.outcome, str)
#
# # test for using multiple raw timeseries loading at a time - using patient wrapper, or center wrapper
# @pytest.mark.usefixture('ieeg_data_fif', 'scalp_data_fif')
# class TestRawMultipleLoader():
#     def test_subj_rawloader(self, ieeg_data_fif):
#         testdatadir, testfilename = ieeg_data_fif
#         loader = SubjectRawLoader(root_dir=testdatadir,
#                                   subjid='ummc001',
#                                   preload=False,
#                                   datatype='ieeg',
#                                   reference='monopolar')
#         loader.read_all_files()
#
#         assert len(loader.datasets) == len(loader.dataset_ids)
#         assert len(loader.datasets) == len(loader.jsonfilepaths)
#
#         # test all datasets
#         datasets = loader.get_results()
