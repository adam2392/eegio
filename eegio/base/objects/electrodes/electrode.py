import collections
import re
from typing import List

from natsort import natsorted

MAX_CONTACTS_IN_ELECTRODE = 20


class Electrode(object):
    def __init__(
        self,
        contact_list: List = None,
        elec_type: str = None,
        wm_contacts: List = None,
        out_contacts: List = None,
        bad_contacts: List = None,
    ):
        """
        Electrode class for a single electrode, with multiple contacts along the SEEG electrode.
        This object handles the regular exp parsing through white matter contacts, out contacts, and
        bad contacts as passed in via user.

        Relies on a hardcoded parameter: MAX_CONTACTS_IN_ELECTRODE
        which will be the maximum number of contacts, the object assumes an electrode can have.

        If an out_contact is found, then all contacts after that up to MAX_CONTACTS_IN_ELECTRODE are
        assumed to be out as well.

        Parameters
        ----------
        contact_list : List
        elec_type : str
        wm_contacts : List
        out_contacts : List
        bad_contacts : List

        Returns
        -------
        A SEEG electrode object.

        """
        if contact_list == None:
            contact_list = []
        if elec_type == None:
            elec_type = "seeg"
        if wm_contacts == None:
            wm_contacts = []
        if out_contacts == None:
            out_contacts = []
        if bad_contacts == None:
            bad_contacts = []

        # keep contact list naturally sorted
        self.contact_list = natsorted(contact_list)
        self.elec_type = elec_type
        self.wm_contacts = wm_contacts
        self.out_contacts = out_contacts
        self.bad_contacts = bad_contacts

        # check all the data for contact list for this electrode
        self._check_data()

        self._process_wm_contacts()
        self._process_out_contacts()

        # do certain things for SEEG electrodes
        if elec_type == "seeg":
            self._process_out_contacts()

        # get the good contacts
        self._get_good_contacts()

    def add_wm_contacts(self, wm_contacts: List):
        self.wm_contacts = wm_contacts
        self._process_wm_contacts()
        # get the good contacts
        self._get_good_contacts()

    def add_out_contacts(self, out_contacts: List):
        self.out_contacts = out_contacts
        self._process_out_contacts()
        # get the good contacts
        self._get_good_contacts()

    def add_bad_contacts(self, bad_contacts: List):
        self.bad_contacts = bad_contacts
        self._process_bad_contacts()
        # get the good contacts
        self._get_good_contacts()

    def _separate_electrode_name_number(self, contact_list):
        contacts = []
        contact_numbers = []
        electrode_names = set()
        for contact in contact_list:
            match = re.match("^([A-Za-z]+[']?)([0-9]+)$", contact)
            if len(match.groups()) != 2:
                raise AttributeError(
                    f"Contact {contact} in contact_list attribute "
                    f"does not split into an electrode name and contact number."
                )
            _elecname, _num = match.groups()
            contacts.append(contact)
            contact_numbers.append(int(_num))
            electrode_names.add(_elecname)
        if len(electrode_names) > 1:
            raise AttributeError(
                f"For this electrode object, you passed in multiple electrodes "
                f"it seems. There were two electrodes found: {electrode_names}."
            )
        return contacts, electrode_names, contact_numbers

    def _check_data(self):
        self.electrode_dict = collections.defaultdict(list)

        # separate electrode name and number
        (
            contacts,
            electrode_names,
            contact_numbers,
        ) = self._separate_electrode_name_number(self.contact_list)

        # create dictionary for the contact list
        self.electrode_dict[electrode_names[0]] = contacts
        self.name = electrode_names[0]

    def _process_out_contacts(self):
        # make sure if contacts ar outside, then everything after is outside
        (
            contacts,
            electrode_names,
            contact_numbers,
        ) = self._separate_electrode_name_number(self.out_contacts)

        """ Doesn't do anything w/ errors yet. Assumes that min out contact -> everything after is outside """
        error_list = []

        # create expected integer range of the contact numbers
        expected_contact_numbers = range(
            min(contact_numbers), MAX_CONTACTS_IN_ELECTRODE + 1
        )

        # go through the contact numbers and see if there are missing
        for idx, _number in enumerate(expected_contact_numbers):
            if idx >= len(contact_numbers):
                break
            if contact_numbers[idx] != _number:
                error_list.append(idx)

    def _process_wm_contacts(self):
        # make sure if contacts ar outside, then everything after is outside
        (
            contacts,
            electrode_names,
            contact_numbers,
        ) = self._separate_electrode_name_number(self.wm_contacts)

    def _process_bad_contacts(self):
        (
            contacts,
            electrode_names,
            contact_numbers,
        ) = self._separate_electrode_name_number(self.bad_contacts)

    def _get_good_contacts(self):
        good_contacts = []

        # create a unionized list of all the contacts to exclude
        self.bad_contacts = list(
            set(self.bad_contacts).union(self.wm_contacts).union(self.out_contacts)
        )

        # go through and set everything
        self.good_contacts = list(set(self.contact_list).difference(self.bad_contacts))

    def __repr__(self):
        return self.good_contacts

    def __str__(self):
        return self.good_contacts

    def __len__(self):
        return len(self.contact_list)
