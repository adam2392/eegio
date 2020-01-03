import random

import numpy as np
import pytest

from eegio.base.objects.electrodes.elecs import Contacts

from eegio.base.utils import ContactsHelper


class TestContacts:
    @pytest.mark.usefixture("contacts")
    def test_contacts_errors(self, contacts):
        """
        Test error and warnings raised by Contacts class.

        :param contacts: (Contacts)
        :return: None
        """
        # no contact coords loaded yet should get a runtime error
        with pytest.warns(UserWarning):
            contact_xyz = contacts.get_contacts_xyz()
        assert contact_xyz is None

        # not passing in correctly lengthed contactlist and coords should result in error
        with pytest.raises(AttributeError):
            contact_xyz = []
            for i in range(len(contacts)):
                test_coord = (random.random(), 1, 0)
                contact_xyz.append(test_coord)

            contactlist = contacts.chanlabels
            with pytest.warns(UserWarning):
                contacts = Contacts(contactlist, contact_xyz[1:])

        with pytest.warns(UserWarning):
            contacts.natsort_contacts()
            contacts.natsort_contacts()

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.usefixture("contacts")
    def test_contacts(self, contacts):
        """
        Test code runs without errors through all functions with dummy data.

        :param contacts: (Contacts)
        :return: None
        """
        contact_xyz = []
        for i in range(len(contacts)):
            test_coord = (random.random(), 1, 0)
            contact_xyz.append(test_coord)
        random_mask = np.random.choice(np.arange(len(contacts)), 6)

        # load xyz coordinates
        print(len(contact_xyz), len(contacts.chanlabels))
        contacts.load_contacts_xyz(contact_xyz, referencespace="CT", scale="mm")

        contacts_xyz = contacts.get_contacts_xyz()
        assert isinstance(contacts_xyz, dict)
        assert len(contacts_xyz.keys()) == len(contacts)

        # assert length of hardcoded electrodes from conftest is 6
        assert len(contacts.electrodes) == 6

        contacts.natsort_contacts()
        contacts.mask_indices(random_mask)

        # generate random mask to get rid of
        random_ch_mask = np.random.choice(contacts.chanlabels, 6)
        contacts.mask_chs(random_ch_mask)

        contact = contacts.chanlabels[0]
        seeg_nghbrs, seeg_nghbrinds = contacts.get_seeg_ngbhrs(contact)

        nghbrs, nghbrinds = contacts.get_contact_ngbhrs(contact)
        assert len(nghbrs) <= len(seeg_nghbrs)
        # print(contacts_xyz)
        # print(contacts.chanlabels)
        # print(contact, nghbrs, seeg_nghbrs)
        contact_xyz = contacts.get_contact_coords(contact)
        assert len(contact_xyz) == 3

        bipres = contacts.set_bipolar()
        assert len(bipres) == 2
        assert isinstance(bipres, tuple)

        bipchs = contacts.chanlabels_bipolar
        bipinds, remaininglabs = bipres

        electrodename = contacts.get_elec(contact)
        assert electrodename is not None
        ch = contacts[1]
        assert isinstance(ch, str)

    @pytest.mark.usefixture("contacts")
    def test_contacts_expansion(self, contacts):
        ch_list = contacts[:]
        ablated_list = ch_list[0:4]
        onset_labels = ch_list[2:]

        bip_chs = ContactsHelper.expand_bipolar_chans(ch_list)
        ablated_list = ContactsHelper.expand_ablated_chans(ch_list)
        onset_labels = ContactsHelper.make_onset_labels_bipolar(ch_list)
