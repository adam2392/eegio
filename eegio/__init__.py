from os.path import dirname, basename, isfile
import glob
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=PendingDeprecationWarning)

from .format.format_eeg_data import run_formatting_eeg
from .format.format_clinical_sheet import format_clinical_sheet
import eegio.base.objects as objects

import eegio.base.utils as utils

modules = glob.glob(dirname(__file__) + "/*.py")
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")
]
name = "eegio"
