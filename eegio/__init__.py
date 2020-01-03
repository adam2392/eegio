"""MNE software for easily interacting with MNE-Python, MNE-BIDS compatible datasets."""
from os.path import dirname, basename, isfile
import glob
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=PendingDeprecationWarning)

import eegio.base.objects as objects
import eegio.base.utils as utils

# modules = glob.glob(dirname(__file__) + "/*.py")
# __all__ = [
#     basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")
# ]
name = "eegio"
__version__ = "0.1.2"
