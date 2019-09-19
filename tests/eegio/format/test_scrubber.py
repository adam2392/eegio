import pytest
import random
from eegio.format.scrubber import ChannelScrub, EventScrub


class TestContacts:
    @pytest.mark.usefixture("rawio")
    def test_channelscrubber(self, rawio):
        """
        Test error and warnings raised by Contacts class.

        :param contacts: (Contacts)
        :return: None
        """
        pass
        # no contact coords loaded yet should get a runtime error
        # with pytest.warns(UserWarning):
        #     contact_xyz = contacts.get_contacts_xyz()
        # assert contact_xyz is None
        #
        # # load xyz coordinates - assert warning
        # # with pytest.warns(UserWarning):
        # #     contacts.load_contacts_xyz(contact_xyz)
        #
        # # not passing in correctly lengthed contactlist and coords should result in error
        # with pytest.raises(AttributeError):
        #     contact_xyz = []
        #     for i in range(len(contacts)):
        #         test_coord = (random.random(), 1, 0)
        #         contact_xyz.append(test_coord)
        #
        #     contactlist = contacts.chanlabels
        #     contacts = Contacts(contactlist, contact_xyz[1:])
        #
        # with pytest.warns(UserWarning):
        #     contacts.natsort_contacts()
        #     contacts.natsort_contacts()

    def test_eventscrubber(self):
        test_eventonsets = []
        test_eventdurations = []
        test_eventkeys = []

        test_eventids = []
        multiplesz = False
        # pass case for finding onset/offset, marker

        pass
