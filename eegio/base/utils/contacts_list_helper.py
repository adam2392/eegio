import re

import numpy as np
from natsort import natsorted


class ContactsHelper:
    @staticmethod
    def expand_bipolar_chans(ch_list):
        if isinstance(ch_list, np.ndarray):
            ch_list = ch_list.tolist()
        if ch_list == []:
            return None

        ch_list = [a.replace("’", "'") for a in ch_list]
        new_list = []
        for string in ch_list:
            if not string.strip():
                continue

            # A1-10
            match = re.match("^([A-Z]+)([0-9]+)-([0-9]+)$", string)
            if match:
                name, fst_idx, last_idx = match.groups()

                new_list.extend([name + str(fst_idx), name + str(last_idx)])

        return new_list

    @staticmethod
    def expand_ablated_chans(ch_list):
        if isinstance(ch_list, np.ndarray):
            ch_list = ch_list.tolist()
        if ch_list == []:
            return None

        ch_list = [a.replace("’", "'") for a in ch_list]
        new_list = []
        for string in ch_list:
            if not string.strip():
                continue

            # A1-10
            match = re.match("^([A-Z]+)([0-9]+)-([0-9]+)$", string)
            if match:
                name, fst_idx, last_idx = match.groups()

                new_list.extend([name + str(fst_idx), name + str(last_idx)])

        return new_list

    @staticmethod
    def make_onset_labels_bipolar(clinonsetlabels):
        clinonsetlabels = list(clinonsetlabels)

        added_ch_names = []
        for ch in clinonsetlabels:
            # A1-10
            match = re.match("^([A-Z]+)([0-9]+)$", ch)
            if match:
                name, fst_idx = match.groups()
            added_ch_names.append(name + str(int(fst_idx) + 1))

        clinonsetlabels.extend(added_ch_names)
        clinonsetlabels = list(set(clinonsetlabels))
        return clinonsetlabels

    @staticmethod
    def get_seeg_ngbhrs(chanlabels, contact: str):
        """
        Helper function to get neighboring contacts for SEEG contacts using regex.

        Parameters
        ----------
        contact : str

        Returns
        -------

        """
        # initialize empty data structures to return
        nghbrcontacts, nghbrinds = [], []

        # get the electrode, and the number for each channel
        elecname, num = re.match("^([A-Za-z]+[']?)([0-9]+)$", contact).groups()

        # find the elecname in rest of electrodes
        elecmaxnum = 0
        elec_numstoinds = dict()
        for jdx in range(len(chanlabels)):
            _elecname, _num = re.match(
                "^([A-Za-z]+[']?)([0-9]+)$", chanlabels[jdx]
            ).groups()
            if elecname == _elecname:
                elecmaxnum = max(elecmaxnum, int(_num))
                elec_numstoinds[_num] = jdx

        # find keys with number above and below number
        elecnumkeys = natsorted(elec_numstoinds.keys())
        elecnumkeys = np.arange(1, elecmaxnum).astype(str).tolist()

        # print(elecnumkeys)

        if num in elecnumkeys:
            currnumind = elecnumkeys.index(num)
            lowerind = max(0, currnumind - 2)
            upperind = min(int(elecnumkeys[-1]), currnumind + 2)

            # print(num, currnumind, lowerind, upperind)

            if lowerind == currnumind:
                lower_nghbrs = np.array([currnumind])
            else:
                lower_nghbrs = np.arange(lowerind, currnumind)

            if currnumind + 1 == upperind:
                upper_nghbrs = np.array([currnumind + 1])
            else:
                upper_nghbrs = np.arange(currnumind + 1, upperind)

            # print(lower_ngbhrs, upper_ngbhrs)
            nghbrinds = np.vstack((lower_nghbrs, upper_nghbrs))
            nghbrcontacts = chanlabels[nghbrinds]

        return nghbrcontacts, nghbrinds

    @staticmethod
    def get_contact_ngbhrs(electrodes, electrodename, contact_name):
        """
        Helper function to return the neighbor contact names, and also the indices in our data structure.

        Parameters
        ----------
        contact_name :

        Returns
        -------

        """
        # initialize empty data structures to return
        nghbrcontacts, nghbrinds = [], []

        electrodecontacts = electrodes[electrodename]

        if contact_name in electrodecontacts:
            contact_index = electrodecontacts.index(contact_name)

            # get the corresponding neighbor indices
            _lowerind = max(contact_index - 1, 0)
            _upperind = min(contact_index + 1, 0)
            nghbrinds = np.vstack(
                (
                    np.arange(_lowerind, contact_index),
                    np.arange(contact_index + 1, _upperind + 1),
                )
            )
            nghbrcontacts = electrodecontacts[nghbrinds]

        return nghbrcontacts, nghbrinds
