import re

import numpy as np

from eegio.base.objects import Contacts


class BaseConverter(object):
    """
    Class for a base data converter. It mainly handles universal text parsing and
    preprocessing of files done to get to our final desired file format (i.e. fif+json file pair)
    per timeseries.

    It allows mainly:
    1. .edf files -> .fif + .json
    2. .edf files -> .mat files

    TODO:
    1. Add conversion from .edf to .hdf files
    2. Add conversion from Nihon Kohden to .edf files
    """

    def __setattr__(self, name, value):
        super(BaseConverter, self).__setattr__(name, value)

    def _reformatchanlabel(self, label):
        """
        Helper function to process a single channel label to make sure it is:

        - lower case
        - removed unnecessary strings (POL, eeg, -ref)
        - removed empty spaces

        :param label: (str) a contact label that may have extra chars, or be improperly cased
        :return: label (str) the reformatted string that is lowercase and w/o spaces
        """
        label = str(label).replace('POL', '').replace(' ', '').lower()
        label = label.replace('eeg', '').replace('-ref', '')

        # replace "Grid" with 'G' label
        label = label.replace('grid', 'g')

        ''' HARD CODED RULES '''
        # for pt17 replace ! with 1
        label = label.replace('!', '1')
        # for pt14 replace ` with ''
        label = label.replace('`', '')
        # label = label.upper()
        return label

    def _addchans_to_raw(self, rawinfo, chanlabels):
        """
        Helper function to add channel labels that are reformatted to the raw info data structure.

        :param rawinfo: (mne.Info) is the mne Python Info data structure that is similar to a dictionary
        :param chanlabels: (list) of strings that are the channel labels for the EEG timeseries
        :return:
        """
        # create a new info object
        rawinfo['ch_names'] = chanlabels

        for item in rawinfo['chs']:
            # process these channel labels
            item['ch_name'] = self._reformatchanlabel(item['ch_name'])

        return rawinfo

    def check_info(self, info):
        for item in info['chs']:
            ch_name = item['ch_name']

            if ch_name not in info['ch_names']:
                print(ch_name)
                print(info['ch_names'])
                # continue
                raise ValueError("{} not in ch names?".format(ch_name))

        for item in info['bads']:
            if item not in info['ch_names']:
                print(item)
                # continue
                raise ValueError("{} not in ch names?".format(item))

    def _loadinfodata(self, rawinfo):
        """
        Helper function to extract relevant data from a MNE.io.Info structure dictionary
        object.

        :param rawinfo: (dict) the raw info object
        :return: None
        """
        # set all dictionary elements in raw info to this class object
        for key, val in rawinfo.items():
            setattr(self, key, val)

        # set samplerate
        self.sfreq = float(self.sfreq)
        self.ch_names = [self._reformatchanlabel(x)
                         for x in self.ch_names]

    def label_channel_types(self):
        """
        Function to load in the channel types. The possibilities are: EEG, STIM, EOG, EKG, Misc.
        that are from MNE-Python.

        We map these to:
        1. bad-non: bad or non-eeg channels
        2. grid: grid channels (1-k*8 contacts)
        3. strip: strip channels (1-6, or 1-8 contacts)
        4. seeg: depth channels inserted (1-8 up to 1-16)

        :return: None
        """

        def remove_letters(s):
            no_digits = []
            # Iterate through the string, adding non-numbers to the no_digits list
            for i in s:
                if i.isdigit():
                    no_digits.append(i)

            # Now join all elements of the list with '',
            # which puts all of the characters together.
            result = ''.join(no_digits)
            return result

        contacts = Contacts(contacts_list=self.ch_names,
                            require_matching=False)

        # create hash dictionary to store label of each channel
        self.channeltypes = {}

        for chanlabel in contacts.chanlabels:
            # get electrode label for this channel
            eleclabel = contacts.get_elec(chanlabel)

            # get rest of electrode labels
            elec_contacts_nums = [int(remove_letters(self.ch_names[ind]))
                                  for ind in contacts.electrodes[eleclabel]]

            if elec_contacts_nums == []:
                self.channeltypes[chanlabel] = 'bad-non'
            elif eleclabel == 'g':
                self.channeltypes[chanlabel] = 'grid'
            elif max(elec_contacts_nums) <= 6:
                self.channeltypes[chanlabel] = 'strip'
            elif max(elec_contacts_nums) > 6 and max(elec_contacts_nums) < 20:
                self.channeltypes[chanlabel] = 'seeg'
            else:
                self.channeltypes[chanlabel] = 'eeg'
                # warnings.warn("Can't determine the channel type for this contact! "
                #               "What should it be? {}".format(chanlabel))

    def find_onsets_offsets(self, samplerate, event_times, event_ids):
        """
        Finds onset and offset seconds and indices based on the event list extracted
        from a timeseries file (.edf).

        :param samplerate:
        :param event_times:
        :param event_ids:
        :return:
        """
        # initialize list of onset/offset seconds
        onset_secs = []
        offset_secs = []
        onsetset = False
        offsetset = False

        offset_sec = None
        onset_sec = None
        onset_ind = None
        offset_ind = None

        if len(event_times) == 0:
            return onset_sec, offset_sec, onset_ind, offset_ind

        # extract the 3 columns from event times
        eventonsets = event_times[:, 0]
        eventdurations = event_times[:, 1]
        eventkey = event_times[:, 2]

        print("Event keys: ", event_times)
        print("Event onset indices:" , eventonsets)

        # onset / offset markers
        onsetmarks = ['onset', 'crise', 'cgtc', 'sz', 'absence']
        offsetmarks = ['offset', 'fin', 'end',
                       'over'
                       ]
        # iterate through the events and assign onset/offset times if avail.
        for name, eventid in event_ids.items():
            name = ','.join(name.lower().split(' '))

            # search for onset markers
            if any(re.search(r'\b{}\b'.format(x), name) for x in onsetmarks):
                if not onsetset:
                    idx = np.where(eventkey == eventid)[0][0]

                    onset_secs = eventonsets[idx].astype(float) // samplerate
                    onsetset = True

                    # continue

            # search for offset markers
            elif any(re.search(r'\b{}\b'.format(x), name) for x in offsetmarks):
                if not offsetset:
                    idx = np.where(eventkey == eventid)[0][0]

                    offset_secs = eventonsets[idx].astype(float) // samplerate
                    offsetset = True

                    # continue

        # set onset/offset times and markers
        try:
            onset_sec = onset_secs
            onset_ind = np.ceil(np.multiply(onset_secs, samplerate))
        except TypeError as e:
            print(e)
            onset_sec = None
            onset_ind = None
        try:
            offset_sec = offset_secs
            offset_ind = np.ceil(np.multiply(offset_secs, samplerate))
        except TypeError as e:
            print(e)
            offset_sec = None
            offset_ind = None

        print(onset_sec, offset_sec)

        if isinstance(onset_sec, list) and len(onset_sec) > 0:
            onset_sec = onset_sec[0]
        if isinstance(offset_sec, list) and len(offset_sec) > 0:
            # print(offset_sec)
            offset_sec = offset_sec[0]

        print(event_times)
        print(event_ids)
        print("Found onset: ", onset_sec)
        print("Found offset: ", offset_sec)
        return onset_sec, offset_sec, onset_ind, offset_ind

    def load_file(self, filepath):
        """
        Abstract method for loading a file. Needs to be implemented by any data converters that will
        load a file to convert.

        :return: None
        """
        raise NotImplementedError(
            "Implement function for loading in file for starting conversion!")
