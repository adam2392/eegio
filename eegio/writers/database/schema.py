class RawMetaSchema(object):
    def __init__(self):
        self.rawschema = {
            "patient_id": str,
            "dataset_id": str,
            "age": float,
            "onset_age": float,
            "gender": str,
            "handedness": str,
            "clinical_center": str,
            "clinical_difficulty": int,
            "clinical_matching": int,
            "bad_channels": list,
            "non_eeg_channels": list,
            "chanxyz": list,
            "chanlabels": list,
            "contact_regs": list,
            "outcome": str,
            "engel_score": float,
            "resected_contacts": list,
            "ablated_contacts": list,
            "clinez_contacts": list,
            "onset_contacts": list,
            "early_spread_contacts": list,
            "late_spread_contacts": list,
            "samplerate": float,
            "lowpass_freq": float,
            "highpass_freq": float,
            "linefreq": float,
            "modality": str,
            "meas_date": float,
            "rawfilename": str,
            "onsetsec": float,
            "offsetsec": float,
            "onsetind": int,
            "offsetind": int,
            "type": str,
            "note": str,
            "events": list,
            "number_chans": int,
        }

    def schemamapcheck(self, metadata):
        pass

    def map_keys(self):
        pass


class MetadataSchema(object):
    def __init__(self):
        self.datasetschema = {
            "patient_id": str,
            "dataset_id": str,
            "age_surgery": float,
            "onset_age": float,
            "gender": str,
            "hand_dominant": str,
            "clinical_center": str,
            "bad_channels": list,
            "non_eeg_channels": list,
            "chanxyz": list,
            "chanlabels": list,
            "contact_regs": list,
            "number_chans": int,
            "outcome": str,
            "engel_score": float,
            "clinical_difficulty": int,
            # 'clinical_matching': int,
            "resected_contacts": list,
            "ablated_contacts": list,
            "seizure_semiology": list,
            "ez_hypo_contacts": list,
            # 'onset_contacts': list,
            # 'early_spread_contacts': list,
            # 'late_spread_contacts': list,
            "winsize": float,
            "stepsize": float,
            "radius": float,
            "stabilizeflag": bool,
            "perturbtype": str,
            "samplerate": float,
            "lowpass_freq": float,
            "highpass_freq": float,
            "linefreq": float,
            "modality": str,
            "reference": str,
            "meas_date": float,
            "rawfilename": str,
            "note": str,
            "onsetsec": float,
            "offsetsec": float,
            "onsetind": int,
            "offsetind": int,
            "samplepoints": list,
        }

    def schemamapcheck(self, metadata, run_again=True):
        # determine type differences in our set of metadata points
        problemkeys = []
        for key in self.datasetschema.keys():
            if key == "stabilizeflag":
                metadata[key] = False
            if not type(metadata[key]) == self.datasetschema[key]:
                problemkeys.append((key, type(metadata[key])))

        if problemkeys and run_again:
            print("Problems with these keys!")
            print(problemkeys)
        else:
            print("Metadata looks okay!")
            # map old keys to new keys once
            self.map_keys(metadata)
            return 1

        # perform type conversion for metadata elements
        for (key, _) in problemkeys:
            try:
                metadata[key] = self.datasetschema[key](metadata[key])
            except Exception as e:
                print(e)

        # allow recursive call to recheck the schema mapping
        self.schemamapcheck(metadata, run_again=False)

    def trimkeys(self, metadata):
        newmetadata = {}
        for key in metadata.keys():
            if key in self.datasetschema.keys():
                newmetadata[key] = metadata[key]
        return newmetadata

    def map_keys(self, metadata):
        mapping = {
            # 'onsetage': 'onset_age',
            # 'clinresectelecs': 'resected_contacts',
            # 'clinezelecs': 'clinez_contacts',
        }

        for oldkey, newkey in mapping.items():
            metadata[newkey] = metadata[oldkey]
            metadata.pop(oldkey)
