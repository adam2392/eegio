class DatasetTester(object):
    """
    A class for testing datasets in the form of fif+json objects. Each dataset will have
    a formatted metadata dictionary object associated with it. This class runs a pipeline of tests
    on that dataset and creates a printable list of attributes that have issues.

    Attributes
    ----------
    metadataobj : dict
        metadata dictionary

    Examples
    --------
    >>> from eegio.dataset_test.dataset_tester import DatasetTester
    # test loading in the actual data from the fif saved place
    >>> loader = iEEGRecording(root_dir=datadir,
    ...                         jsonfilepath=jsonpath,
    ...                          preload=False)
    >>> ieegts = loader.loadpipeline()
    >>> metadata = ieegts.get_metadata()
    >>> tester = DatasetTester()
    >>> # test the metadata for this dataset
    >>> tester.load_datadict(metadata)
    >>> tester.test_pipeline()
    >>> test_results = tester.get_test_results()
    """

    def __init__(self, modality, metadataobj=None):
        self.data = metadataobj
        self.modality = modality

        # keep a data structure of all the problems in dataset found
        self.problems = []

    def get_test_results(self):
        """
        Getter method for getting the results of testing the dataset. One should
        print the results to see which dataset's attributes failed the dataset test.

        :return: self.problems (list)
        """
        return self.problems

    def load_datadict(self, metadataobj):
        """
        Loader method for passing a metadata dictionary object to the DatasetTester class object.

        :param metadataobj: (dict)
        :return: None
        """
        self.data = metadataobj
        self.problems = []

    def test_pipeline(self):
        """
        Main function to call that runs tests in a sequence of class method calls.

        :return: None
        """
        print("Checking dataset type!")
        self.check_dataset_type()

        print("Checking clinical outcomes!")
        self.check_clinical_outcomes()

        print("Checking onset/offset!")
        self.check_onset_offset()

        print("Checking cez labels!")
        if self.modality == 'ieeg':
            self.check_cez_labels_ieeg()
        elif self.modality == 'scalp':
            self.check_cez_labels_scalp()

    def check_dataset_type(self):
        """
        Function to test the dataset's types. We have only seizures, and interictal
        possibilities.

        TODO:
        - add awake/asleep partition

        :return: None
        """
        keys = {
            "type": ['sz', 'ii']
        }
        for key, possibilities in keys.items():
            if key in self.data.keys():
                if any(possibility in self.data[key] for possibility in possibilities):
                    continue
                else:
                    self.problems.append((key, self.data[key]))
            else:
                self.problems.append((key, None))

    def check_modality(self):
        """
        Function to test the dataset's modality. We have the possibilities of:
        - seeg
        - ecog
        - scalp
        - ieeg (the most general case of seeg and ecog)

        These separate modality names, help corresponding functions interpret
        different channesl differently.

        :return: None
        """
        keys = {
            'modality': ['seeg', 'ecog', 'scalp', 'ieeg']
        }
        for key, possibilities in keys.items():
            if key in self.data.keys():
                if self.data[key] in possibilities:
                    continue
                else:
                    self.problems.append((key, self.data[key]))
            else:
                self.problems.append((key, None))

    def check_clinical_outcomes(self):
        """
        Function to check clinical outcomes of a dataset. Every patient/dataset
        has a surgical outcome, engel_score and clinical_difficulty associated with it.

        Note that if an outcome was no resection (i.e. nr), then the corresponding
        engel_score should be -1.

        :return: None
        """
        keys = {
            'outcome': ['s', 'f', 'nr'],
            'engel_score': [-1, 1, 2, 3, 4],
            'clinical_difficulty': [1, 2, 3, 4],
        }

        for key, possibilities in keys.items():
            if key in self.data.keys():
                if self.data[key] in possibilities:
                    continue
                else:
                    self.problems.append((key, self.data[key]))
            else:
                self.problems.append((key, None))

    def check_onset_offset(self):
        """
        Function to check the onset/offset times of a dataset. Every seizure dataset
        should have a marked eeg_onset and eeg_offset from clinical annotations.

        TODO:
        - add support for clinical_onset

        :return: None
        """
        keys = {
            'onset': [None],
            'termination': [None],
        }

        if self.data['type'] == 'ii':
            return

        if self.data['onset'] == []:
            self.problems.append(('onset', []))

        if self.data['termination'] == []:
            self.problems.append(('termination', []))

        try:
            if self.data['onset'] >= self.data['termination']:
                self.problems.append(('onset', self.data['onset']))
                self.problems.append(('termination', self.data['termination']))
        except TypeError as e:
            print(e)

    def check_cez_labels_scalp(self):
        """
        Function to check the clinically annotated EZ labels at the contact level.
        It checks for the:
        - ez_hypo_contacts that can be different per dataset
        - ablated_contacts/resected_contacts that will be the same for every dataset for a specific patient
        - seizure_semiology that outlines the onset and spread contacts annotated per dataset

        :return: None
        """
        keys = {
            "cezlobe": [],
            # "implantation_distribution": [],
        }

        for key, possibilities in keys.items():
            if key in self.data.keys():
                if isinstance(self.data[key], list):
                    continue
                else:
                    self.problems.append((key, self.data[key]))
            else:
                self.problems.append((key, None))

        not_empty = False
        for key in keys.keys():
            if self.data[key] != []:
                not_empty = True
        if not not_empty:
            self.problems.append((keys))

    def check_cez_labels_ieeg(self):
        """
        Function to check the clinically annotated EZ labels at the contact level.
        It checks for the:
        - ez_hypo_contacts that can be different per dataset
        - ablated_contacts/resected_contacts that will be the same for every dataset for a specific patient
        - seizure_semiology that outlines the onset and spread contacts annotated per dataset

        :return: None
        """
        keys = {
            "ez_hypo_contacts": [],
            "resected_contacts": [],
            "ablated_contacts": [],
            "seizure_semiology": [],
        }

        for key, possibilities in keys.items():
            if key in self.data.keys():
                if isinstance(self.data[key], list):
                    continue
                else:
                    self.problems.append((key, self.data[key]))
            else:
                self.problems.append((key, None))

        not_empty = False
        for key in keys.keys():
            if self.data[key] != []:
                not_empty = True
        if not not_empty:
            self.problems.append((keys))

        self._check_cez_labels_in_chans()

    def _check_cez_labels_in_chans(self):
        """
        Helper function to check the cez labels for contact level.

        :return: None
        """
        keys = {
            "ez_hypo_contacts": [],
            "resected_contacts": [],
            "ablated_contacts": [],
        }
        chanlabels = self.data['chanlabels']

        for key in keys.keys():
            keyitems = self.data[key]

            for item in keyitems:
                if item not in chanlabels and item not in self.data['bad_channels'] and item not in self.data[
                    'non_eeg_channels']:
                    self.problems.append((key, item))

        keyitems = [item for sublist in self.data['seizure_semiology']
                    for item in sublist]
        for item in keyitems:
            if item not in chanlabels and item not in self.data['bad_channels'] and item not in self.data[
                'non_eeg_channels']:
                self.problems.append(('seizure_semiology', item))

    def check_cez_brain_regionlabels(self):
        """
        Function to check the brain regions labeled as suspected epileptogenic.

        :return: None
        """
        pass
