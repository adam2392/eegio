ACCEPTED_EEG_MODALITIES = [
    "scalp",
    "ecog",
    "seeg",
    "ieeg",
    "eeg",
]  # EEG modalities that EEGIO supports
DATE_MODIFIED_KEY = "date_modified"  # dataset modified tracker keyname

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
