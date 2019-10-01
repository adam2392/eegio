import collections
import re
import warnings
from typing import List

import numpy as np
from natsort import natsorted, order_by_index, index_natsorted


def reinitialize_datastructure(f):
    f()
    f._initialize_datastructs()
    return f


class Contacts(object):
    """
    Class wrapper for a set of contacts.

    Allows the user to:
     - load contacts and if available, their corresponding T1 brain xyz coordinates.
     - mask contacts that we don't want to consider
     - set a reference scheme (monopolar, common_average, bipolar, customized)

    Note that if user sets a reference scheme, then it only simply relabels the channels and applies some metadata. Any
    dataset relying on these specific ordering of contacts, should apply a reference function to their dataset.

    Attributes
    ----------
    contacts_list : (list)
        The list of contacts and their names. (e.g. [l1, l2, l3, a'1, a'2])

    contacts_xyz : (list)
        The list of xyz tuples, which are by default in mm

    referencespace: (str)
        The space you obtained contacts coordinates from. E.g. head, CT, MRI

    scale: (str)
        Scale that the xyz coordinates are on. E.g. mm, cm, m

    require_matching: (bool)
        The boolean that determines if some quality check on the names should be imposed.

    Examples
    --------
    >>> from eegio.base.objects.elecs import Contacts
    >>> contacts_list = ['a1', 'a2', 'a3', 'b1', 'b2', 'b3']
    >>> chan_xyz = [[0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,2,0], [0,3, 0]]
    >>> contacts = Contacts(contacts_list, chan_xyz)
    >>> print("These are the contacts: ", contacts)
    """

    # regex patterns for monopolar, bipolar contact labeling
    contact_single_regex = re.compile("^([A-Za-z]+[']?)([0-9]+)$")
    contact_pair_regex_1 = re.compile("^([A-Za-z]+[']?)([0-9]+)-([0-9]+)$")
    contact_pair_regex_2 = re.compile(
        "^([A-Za-z]+[']?)([0-9]+)-([A-Za-z]+[']?)([0-9]+)$"
    )

    def __init__(
        self,
        contacts_list: List = [],
        contacts_xyz: List = [],
        referencespace: str = None,
        scale: str = None,
        require_matching: bool = True,
    ):
        self.require_matching = require_matching
        self.chanlabels = np.array([x.lower() for x in contacts_list])

        if contacts_xyz is not None and referencespace == None:
            warnings.warn(
                "referencespace should be set explicitly if you are passing in contact coordinates. "
                "Defaulting to None. "
            )
        if contacts_xyz is not None and scale == None:
            warnings.warn(
                "scale should be set explicitly if you are passing in contact coordinates. "
                "Defaulting to mm."
            )
            scale = "mm"
        if contacts_xyz and (len(contacts_list) != len(contacts_xyz)):
            raise AttributeError(
                "Length of contacts_list and contacts_xyz should be the same. "
                "Length was {} and {}.".format(len(contacts_list), len(contacts_xyz))
            )

        # set class data
        self.xyz = contacts_xyz
        self.referencespace = referencespace
        self.scale = scale
        self.natinds = None

        self._initialize_contacts_xyz_dict()

        # initialize data structures
        self._initialize_datastructs()

    def __str__(self):
        return ",".join(self.chanlabels)

    def __repr__(self):
        return self.chanlabels

    def __len__(self):
        return len(self.chanlabels)

    def __getitem__(self, given):
        if isinstance(given, slice):
            # do your handling for a slice object:
            # print(given.start, given.stop, given.step)
            return self.chanlabels[given.start : given.stop : given.step]
        else:
            # Do your handling for a plain index
            # print(given)
            return self.chanlabels[given]

    def _initialize_datastructs(self):
        """
        Helper function to initialize an electrodes dictionary for storing contacts belonging to the same electrode.

        Assumes that all different electrodes have a starting different lettering (e.g. A1, A2, A3 are all from the same
        electrode).

        :return:
        """
        self.electrodes = collections.defaultdict(list)
        for i, name in enumerate(self.chanlabels):
            match = self.contact_single_regex.match(name)
            if match is None:
                if self.require_matching:
                    raise ValueError(
                        f"Unexpected contact name {name}. "
                        f"If you are using scalp electrodes, then pass 'require_matching=False' "
                        f"to prevent regular expression checking."
                    )
                else:
                    continue

            elec_name, _ = match.groups()
            if elec_name not in self.electrodes:
                self.electrodes[elec_name] = []
            self.electrodes[elec_name].append(i)

    def _initialize_contacts_xyz_dict(self):
        self.name_to_xyz = dict(zip(self.chanlabels, self.xyz))

    def load_contacts_xyz(self, contacts_xyz, referencespace=None, scale="mm"):
        """
        Function to help load in contact coordinates.

        :param contacts_xyz:
        :return:
        """
        if referencespace == None:
            warnings.warn(
                "No referencespace passed in. It is set as None. This is ideally passed in "
                "to allow user to determine where the contacts were localized."
            )

        self.xyz = contacts_xyz
        self.referencespace = referencespace
        self.scale = scale
        self._initialize_contacts_xyz_dict()

    def get_contacts_xyz(self):
        """
        Helper function to return the contacts and their xyz coordinates.

        :return: (dict) contacts and their xyz tuples or
                (None) if no xyz
        """
        if self.xyz and (len(self.xyz) == len(self.chanlabels)):
            return {chanlabel: xyz for chanlabel, xyz in zip(self.chanlabels, self.xyz)}

        warnings.warn(
            "Contacts xyz coordinates are not yet loaded. See documentation "
            "for more details on how to load in the coordinates. This will return None."
        )
        return None

    def natsort_contacts(self):
        """
        Naturally sort the contacts. Keeps the applied indices in self.natinds

        :return:
        """
        if self.natinds == None:
            self.natinds = index_natsorted(self.chanlabels)
            self.chanlabels = np.array(order_by_index(self.chanlabels, self.natinds))
        else:
            warnings.warn(
                "Already naturally sorted contacts! Extract channel labels naturally sorted by calling "
                "chanlabels, and apply ordering to other channel level data with natinds."
            )
        return self.natinds

    def mask_contact_indices(self, mask_inds):
        """
        Helper function to mask out certain indices.

        :param mask_inds:
        :return:
        """
        if max(mask_inds) > len(self.chanlabels):
            warnings.warn(
                "You passed in masking indices not in the length of the channel labels. "
                "Length of channel labels is: {}".format(len(self.chanlabels))
            )

        keepinds = [i for i in range(len(self.chanlabels)) if i not in mask_inds]
        self.chanlabels = self.chanlabels[keepinds]
        if self.xyz:
            self.xyz = [self.xyz[i] for i in keepinds]
        return keepinds

    def mask_contacts(self, contacts):
        """
        Helper function to make out certain contacts by name.

        :param contacts:
        :type contacts:
        :return:
        :rtype:
        """
        if any(x not in self.chanlabels for x in contacts):
            raise ValueError(
                "The contacts you passed to mask, were not inside. You passed"
                f"{contacts}. The contacts available are: {self.chanlabels}."
            )

        keepinds = [i for i, ch in enumerate(self.chanlabels) if ch not in contacts]
        self.chanlabels = self.chanlabels[keepinds]
        if self.xyz:
            self.xyz = [self.xyz[i] for i in keepinds]
        return keepinds

    def get_seeg_ngbhrs(self, contact: str):
        """
        Helper function to get neighboring contacts for SEEG contacts.

        :param contact:
        :param chanlist:
        :return:
        """
        # initialize empty data structures to return
        nghbrcontacts, nghbrinds = [], []

        # get the electrode, and the number for each channel
        elecname, num = re.match("^([A-Za-z]+[']?)([0-9]+)$", contact).groups()

        # find the elecname in rest of electrodes
        elecmaxnum = 0
        elec_numstoinds = dict()
        for jdx in range(len(self.chanlabels)):
            _elecname, _num = re.match(
                "^([A-Za-z]+[']?)([0-9]+)$", self.chanlabels[jdx]
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
            nghbrcontacts = self.chanlabels[nghbrinds]

        return nghbrcontacts, nghbrinds

    def get_contact_ngbhrs(self, contact_name):
        """
        Helper function to return the neighbor contact names, and also the indices in our data structure.

        :param contact_name: (str) the contact name (e.g. A1, A2)
        :return: (list of str) contact neighbors (e.g. A1, A3), and
                (list of int) contact neighbor indices
        """
        # initialize empty data structures to return
        nghbrcontacts, nghbrinds = [], []

        match = self.contact_single_regex.match(contact_name)
        if match is None:
            return None
        electrode = match.groups()[0]
        electrodecontacts = self.electrodes[electrode]

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

    def set_bipolar(self):
        """
        Create bipolar pairing from channel labeling.

        :return: (list) bipolar indices
            or   (list, list) bipolar indices and any remaining labels that weren't found.
        """
        # set list to hold new labeling and the corresponding bipolar indices
        self.chanlabels_bipolar = []
        self.bipolar_inds = []

        remaining_labels = np.array([])
        n = len(self.chanlabels)

        # zip through all label indices
        for inds in zip(np.r_[: n - 1], np.r_[1:n]):
            # get the names for all the channels
            names = [self.chanlabels[ind] for ind in inds]

            # get the electrode, and the number for each channel
            elec0, num0 = re.match("^([A-Za-z]+[']?)([0-9]+)$", names[0]).groups()
            elec1, num1 = re.match("^([A-Za-z]+[']?)([0-9]+)$", names[1]).groups()

            # if electrode name matches, and the number are off by 1, then apply bipolar
            if elec0 == elec1 and abs(int(num0) - int(num1)) == 1:
                name = "%s%s-%s" % (elec0, num1, num0)
                self.chanlabels_bipolar.append(name)
                self.bipolar_inds.append([inds[1], inds[0]])

        # convert to numpy arrays and convert the raw data
        self.bipolar_inds = np.array(self.bipolar_inds, dtype=int)

        if remaining_labels.size == 0:
            return self.bipolar_inds, remaining_labels
        else:
            return self.bipolar_inds, remaining_labels

    def get_elec(self, contact_name):
        match = self.contact_single_regex.match(contact_name)
        if match is None:
            return None
        return match.groups()[0]

    def get_contact_coords(self, contact_name):
        """
        Function to get the coordinates of a specified contact or contact pair. Allowed formats are:

            A1            : Single contact.
            A1-2 or A1-A2 : Contact pair. The indices must be adjacent.


        Examples:

        >>> np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
        >>> contacts = Contacts(["A1", "A2"], [(0.0 0.0 1.0), (0.0 0.0 2.0)])
        >>> contacts.get_coords("A1")
        array([0.0, 0.0, 1.0])
        >>> contacts.get_coords("A1-2")
        array([0.0, 0.0, 1.5])
        >>> contacts.get_coords("A2-A1")
        array([0.0, 0.0, 1.5])
        """

        match = self.contact_single_regex.match(contact_name)
        if match is not None:
            return self.name_to_xyz[contact_name]

        match = self.contact_pair_regex_1.match(contact_name)
        if match is not None:
            assert abs(int(match.group(2)) - int(match.group(3))) == 1
            contact1 = match.group(1) + match.group(2)
            contact2 = match.group(1) + match.group(3)
            return (self.name_to_xyz[contact1] + self.name_to_xyz[contact2]) / 2.0

        match = self.contact_pair_regex_2.match(contact_name)
        if match is not None:
            assert match.group(1) == match.group(3)
            assert abs(int(match.group(2)) - int(match.group(4))) == 1
            contact1 = match.group(1) + match.group(2)
            contact2 = match.group(3) + match.group(4)
            return (self.name_to_xyz[contact1] + self.name_to_xyz[contact2]) / 2.0

        raise ValueError(
            "Given name '%s' does not follow any expected pattern." % contact_name
        )

    @classmethod
    def expand_bipolar_chans(self, ch_list):
        if ch_list == []:
            return None

        ch_list = [a.replace("’", "'") for a in ch_list]
        new_list = []
        for string in ch_list:
            if not string.strip():
                continue

            # A1-10
            match = re.match("^([a-z]+)([0-9]+)-([0-9]+)$", string)
            if match:
                name, fst_idx, last_idx = match.groups()

                new_list.extend([name + str(fst_idx), name + str(last_idx)])

        return new_list

    @classmethod
    def expand_ablated_chans(self, ch_list):
        if ch_list == []:
            return None

        ch_list = [a.replace("’", "'") for a in ch_list]
        new_list = []
        for string in ch_list:
            if not string.strip():
                continue

            # A1-10
            match = re.match("^([a-z]+)([0-9]+)-([0-9]+)$", string)
            if match:
                name, fst_idx, last_idx = match.groups()

                new_list.extend([name + str(fst_idx), name + str(last_idx)])

        return new_list

    @classmethod
    def make_onset_labels_bipolar(self, clinonsetlabels):
        clinonsetlabels = list(clinonsetlabels)

        added_ch_names = []
        for ch in clinonsetlabels:
            # A1-10
            match = re.match("^([a-z]+)([0-9]+)$", ch)
            if match:
                name, fst_idx = match.groups()
            added_ch_names.append(name + str(int(fst_idx) + 1))

        clinonsetlabels.extend(added_ch_names)
        clinonsetlabels = list(set(clinonsetlabels))
        return clinonsetlabels
