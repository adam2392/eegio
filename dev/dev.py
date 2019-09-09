class EEGTs(EEGTimeSeries):
    """
    The class object for our scalp EEG time series data.

    All time series are assumed to be in [C x T] shape and use the Contacts
    data structure to handle all contact level functionality.

    Attributes
    ----------
    mat : (np.ndarray)
        The time series EEG timeseries that is NxT

    metadata : (dict)
        The dictionary metadata object.

    Notes
    -----

    Examples
    --------
    >>> from eegio.objects.timeseries.eegts_object import EEGTs
    >>> rawdata = np.random.rand((80,100))
    >>> metadata = dict()
    >>> eegts = EEGTs(rawdata, metadata)
    """

    def __init__(self, mat, metadata):
        super(EEGTs, self).__init__(mat, metadata)

    @property
    def montage(self):
        return self.metadata['montage']

    def apply_montage(self):
        """
        Function to apply a predefined scalp EEG montage to the timeseries.

        :return: mne_rawstruct (mne.Io.Raw) the Raw data structure created from a specific montage.
        """
        montage = self.montage
        if self.montage is None:
            montage = 'standard_1020'

        # create a mne raw data structure with the montage applied
        mne_rawstruct = create_mne_topostruct_from_numpy(self.mat,
                                                         self.chanlabels,
                                                         self.samplerate,
                                                         montage=montage)
        return mne_rawstruct

    def set_bipolar(self, chanlabels=[]):
        """
        Function to apply a bipolar reference scheme to the EEG time series timeseries.

        TODO:
            Implement function

        :param chanlabels: (list) a list of channel labels
        :return: None
        """
        raise RuntimeError("It is not recommended to set bipolar contacts"
                           "on scalp EEG! Should use for SEEG and ECoG.")

    def create_raw_struct(self, remove=True):
        """
        Function to create a mne.io.Raw data structure from the EEGTs timeseries and corresponding metadata (i.e.
        channel labels, samplerate, montage).

        :param remove:
        :return:
        """
        eegts = self.get_data()
        chanlabels = self.chanlabels
        samplerate = self.samplerate

        # gets the best montage
        best_montage = self.get_best_matching_montage(chanlabels)

        # gets channel indices to keep
        montage_chs = chanlabels
        montage_data = eegts
        if remove:
            montage_inds = self.get_montage_channel_indices(
                best_montage, chanlabels)
            montage_chs = chanlabels[montage_inds]
            other_inds = [idx for idx, ch in enumerate(
                chanlabels) if ch not in montage_chs]
            montage_data = eegts[montage_inds, :]

            print("Removed these channels: ", chanlabels[other_inds])

        mne_rawstruct = create_mne_topostruct_from_numpy(
            montage_data, montage_chs, samplerate, montage=best_montage)
        return mne_rawstruct

    def get_montage_channel_indices(self, montage_name, chanlabels):
        """
        Function to get channel indices inside "chanlabels" for a specific montage_name.

        :param montage_name: (str) a predefined montage name from MNE-Python
        :param chanlabels: (list) a list of the channel string labels
        :return: (list) a list of indices for the montage-included channnels
        """
        # read in montage and strip channel labels not in montage
        montage = mne.channels.read_montage(montage_name)
        montage_chs = [ch.lower() for ch in montage.ch_names]
        # get indices to keep
        to_keep_inds = [idx for idx, ch in enumerate(
            chanlabels) if ch in montage_chs]
        return to_keep_inds

    def get_best_matching_montage(self, chanlabels):
        """
        Get the best matching montage with respect to the list of passed in channel label names.

        :param chanlabels: (list) channel label strings to find a matching montage for
        :return: best_montage (str) the name of the montage
        """
        montages = mne.channels.get_builtin_montages()
        best_montage = None
        best_montage_score = 0

        for montage_name in montages:
            # read in standardized montage
            montage = mne.channels.read_montage(montage_name)

            # get the channels and score for this montage wrt channels
            montage_score = 0
            montage_chs = [ch.lower() for ch in montage.ch_names]

            # score this montage
            montage_score = len([ch for ch in chanlabels if ch in montage_chs])
            if montage_score > best_montage_score:
                best_montage = montage_name
                best_montage_score = montage_score
        return best_montage


class SEEGTs(EEGTimeSeries):
    """
    The class object for our SEEG time series data.

    All time series are assumed to be in [C x T] shape and use the Contacts
    data structure to handle all contact level functionality.

    Attributes
    ----------
    mat : (np.ndarray)
        The time series EEG timeseries that is NxT

    metadata : (dict)
        The dictionary metadata object.

    Notes
    -----

    Examples
    --------
    >>> from eegio.objects.timeseries.eegts_object import SEEGTs
    >>> rawdata = np.random.rand((80,100))
    >>> metadata = dict()
    >>> eegts = SEEGTs(rawdata, metadata)
    """

    def __init__(self, mat, metadata):
        super(SEEGTs, self).__init__(mat, metadata)

    def __str__(self):
        return "{} {} SEEG mat ({}) " \
               "{} seconds".format(
                   self.patient_id, self.dataset_id, self.mat.shape, self.len_secs)

    @property
    def cezcontacts(self):
        return self.metadata['ez_hypo_contacts']

    @property
    def channel_semiology(self):
        return self.metadata['seizure_semiology']

    @property
    def ablationcontacts(self):
        return self.metadata['ablated_contacts']

    @property
    def resectedcontacts(self):
        return self.metadata['resected_contacts']

    @property
    def wm_contacts(self):
        if 'wm_contacts' in self.metadata.keys():
            return [x.lower() for x in self.metadata['wm_contacts']]
        return []

    def remove_wmcontacts(self):
        """
        Removes white matter contacts from depth electrode recordings.

        :return: None
        """
        wm_contacts = self.wm_contacts
        keepinds = []
        for i, contact in enumerate(self.chanlabels):
            if contact not in wm_contacts:
                keepinds.append(i)
        self.contacts.mask_contact_indices(keepinds)
        self.mat = self.mat[keepinds, :]
        print("Removed wm contacts: ", self.chanlabels.shape,
              self.mat.shape, flush=True)

    def get_hemisphere_contacts(self, hemisphere):
        """
        Function to return the contact labels that are in a specific hemisphere of the brain.

        :param hemisphere: (str) a string denoting either 'r' (right), or 'l' (left) hemisphere of the brain.
        :return: contacts (list) a list of contact labels
        """
        if hemisphere == 'r':
            contacts = [ch for idx, ch in enumerate(
                self.chanlabels) if "'" not in ch]
        elif hemisphere == 'l':
            contacts = [ch for idx, ch in enumerate(
                self.chanlabels) if "'" in ch]
        else:
            raise ValueError(
                "The only possible values for hemisphere are 'r' and 'l'. You passed in {}".format(hemisphere))
        return contacts


class ECOGTs(EEGTimeSeries):
    """
    The class object for our ECOG time series data.

    All time series are assumed to be in [C x T] shape and use the Contacts
    data structure to handle all contact level functionality.

    Attributes
    ----------
    mat : (np.ndarray)
        The time series EEG timeseries that is NxT

    metadata : (dict)
        The dictionary metadata object.

    Notes
    -----

    Examples
    --------
    >>> from eegio.objects.timeseries.eegts_object import ECOGTs
    >>> rawdata = np.random.rand((80,100))
    >>> metadata = dict()
    >>> eegts = ECOGTs(rawdata, metadata)
    """

    def __init__(self, mat, metadata):
        super(ECOGTs, self).__init__(mat, metadata)

    @property
    def cezcontacts(self):
        return self.metadata['ez_hypo_contacts']

    @property
    def channel_semiology(self):
        return self.metadata['seizure_semiology']

    @property
    def resectedcontacts(self):
        return self.metadata['resected_contacts']

    def __str__(self):
        return "{} {} ECoG mat ({}) " \
               "{} seconds".format(
                   self.patient_id, self.dataset_id, self.mat.shape, self.len_secs)

    def get_grid_contacts(self):
        """
        Function to find and return all the grid contacts in a set of channel labels for ECOGTs.

        TODO:
            - come up with algorithm to separate by electrodes. E.g. can be many grids on a patient.

        :return: grid_contacts (list) a list of the contacts that are grids.
        """
        grid_contacts = [ch for idx, ch in enumerate(
            self.chanlabels) if ch.startswith('g')]
        return grid_contacts

    # def get_metadata(self):
    #     return self.metadata
    #
    # def get_reference_type(self):
    #     return self.reference

    def extract_metadata(self):
        self.samplerate = self.metadata['samplerate']

        if 'patient_id' in self.metadata.keys():
            self.patient_id = self.metadata['patient_id']
            self.dataset_id = self.metadata['dataset_id']
        else:
            self.patient_id = None
            self.dataset_id = None

        self.contacts_list = self.metadata['chanlabels']
        self.linefreq = self.metadata['linefreq']

        if 'clinical_center' in self.metadata.keys():
            self.clinical_center = self.metadata['clinical_center']
        else:
            self.clinical_center = None

        # self.cezlobe = self.metadata['cezlobe']
        if 'outcome' in self.metadata.keys():
            self.outcome = self.metadata['outcome']
        else:
            self.outcome = None
        # self.clinonsetlabels = self.metadata['ez_hypo_contacts']
        # self.semiology = self.metadata['seizure_semiology']
        # self.ablationlabels = self.metadata['ablated_contacts']
        try:
            self.clinicaldifficulty = self.metadata['clinical_difficulty']
        except:
            self.clinicaldifficulty = -1

        try:
            self.clinicalmatching = self.metadata['clinical_match']
        except:
            self.clinicalmatching = -1

        try:
            self.modality = self.metadata['modality']
        except:
            self.modality = None

        try:
            self.cezlobe = self.metadata['cezlobe']
        except:
            self.cezlobe = []

        # self.modality = self.metadata['modality']
        try:
            self.outcome = self.metadata['outcome']
        except:
            self.outcome = ''

        try:
            self.clinicaldifficulty = self.metadata['clinical_difficulty']
        except:
            self.clinicaldifficulty = ''

        try:
            self.clinicalmatching = self.metadata['clinical_match']
        except:
            self.clinicalmatching = ''

        try:
            # comment out
            self.modality = self.metadata['modality']
        except Exception as e:
            self.metadata['modality'] = 'ieeg'
            self.modality = self.metadata['modality']
            print("Error in extracting metadata: ", e)

        self.onsetind = self.metadata['onsetind']
        self.offsetind = self.metadata['offsetind']

        # convert channel labels into a Contacts data struct
        self.contacts = Contacts(self.contacts_list, require_matching=False)

    def create_raw_struct(self, remove=True):
        eegts = self.get_data()
        chanlabels = self.chanlabels
        metadata = self.get_metadata()
        samplerate = self.samplerate

        # if montage not in mne.
        # create the info struct
        info = mne.create_info(ch_names=chanlabels.tolist(), ch_types=[
            'eeg'] * len(chanlabels), sfreq=samplerate)
        # create the raw object
        mne_rawstruct = mne.io.RawArray(data=eegts, info=info)
        return mne_rawstruct
