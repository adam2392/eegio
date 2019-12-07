import os

ACCEPTED_EEG_MODALITIES = [
    "scalp",
    "ecog",
    "seeg",
    "ieeg",
    "eeg",
]  # EEG modalities that EEGIO supports
DATE_MODIFIED_KEY = "date_modified"  # dataset modified tracker keyname

NON_EEG_MARKERS = [
    "dc",
    "ekg",
    "ref",
    "emg",
    "ecg",
    "events",
    "markers",
    "mark",
    "sti014",
    "stim",
    "sti",
    "rfc",
]

BAD_MARKERS = ["$", "FZ", "GZ", "DC", "STI"]

""" Clinical excel sheet configuration """
COLS_TO_REGEXP_EXPAND = [
    "bad_contacts",
    "wm_contacts",
    "out_contacts",
    "soz_contacts",
]  # clinical excel sheet columns to support regular exp expansion

# basic storage units in bytes
MB = 1e6
GB = 1e9


EEGIO_DIR = os.path.normpath(os.path.join(os.getcwd(), "../.."))
TEST_DATA_DIR = os.path.join(EEGIO_DIR, "data", "ieeg_ecog_test.edf")
