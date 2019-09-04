import pytest
import random
import numpy as np
from eegio.base.objects.dataset.eegts_object import EEGTimeSeries

@pytest.mark.usefixture('contacts')
class TestContacts():
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
        random_ch_mask = np.random.choice(contacts, 6)

        # load xyz coordinates
        contacts.load_contacts_xyz(contact_xyz, referencespace="CT", scale="mm")

        contacts_xyz = contacts.get_contacts_xyz()
        assert isinstance(contact_xyz, dict) and len(contact_xyz.keys()) == len(contacts)

        # assert length of hardcoded electrodes from conftest is 6
        assert len(contacts.electrodes) == 6

        contacts.natsort_contacts()
        contacts.mask_contact_indices(random_mask)
        contacts.mask_contacts(random_ch_mask)

        contact = contacts.chanlabels[0]
        nghbrs, nghbrinds = contacts.get_contact_ngbhrs(contact)
        seeg_nghbrs, seeg_nghbrinds = contacts.get_seeg_ngbhrs(contact)
        assert len(nghbrs) <= len(seeg_nghbrs)

        contact_xyz = contacts.get_contact_coords(contact)
        assert len(contact_xyz) == 3

        bipres = contacts.set_bipolar()
        assert len(bipres) == 2 and isinstance(bipres, tuple)

        bipchs = contacts.chanlabels_bipolar
        bipinds, remaininglabs = bipres

        electrodename = contacts.get_elec(contact)
        assert electrodename is not None

    def test_contacts_errors(self, contacts):
        """
        Test error and warnings raised by Contacts class.

        :param contacts: (Contacts)
        :return: None
        """
        contact_xyz = []
        for i in range(len(contacts)):
            test_coord = (random.random(), 1, 0)
            contact_xyz.append(test_coord)
        random_mask = np.random.choice(np.arange(len(contacts)), 6)
        random_ch_mask = np.random.choice(contacts, 6)

        # no contact coords loaded yet should get a runtime error
        with pytest.raises(RuntimeError):
            contact_xyz = contacts.get_contacts_xyz()

        # load xyz coordinates - assert warning
        with pytest.warns("Should run a warning!"):
            contacts.load_contacts_xyz(contact_xyz)

        # not passing in correctly lengthed contactlist and coords should result in error
        with pytest.raises(AttributeError):
            contactlist = contacts.chanlabels
            contacts = Contacts(contactlist, contact_xyz[1:])

        with pytest.warns():
            contacts.natsort_contacts()
            contacts.natsort_contacts()
