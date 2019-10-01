from os.path import dirname, basename, isfile
import glob

# from .format.format_eeg_data import FormatEEGData as format_eegdata
# from .format.format_clinical_sheet import FormatClinicalSheet as format_clinical_sheet

modules = glob.glob(dirname(__file__) + "/*.py")
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")
]
name = "eegio"
