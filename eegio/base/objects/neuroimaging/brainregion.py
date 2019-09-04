import collections

import scipy


class RegionToLobeMapping():
    """
    Class wrapper to map brain regions to a lobe based on two dictionaries:
    1. region -> lobe index
    2. lobe index -> lobe

    """

    def __init__(self, region_to_index, index_to_lobe):
        self.region_to_index = region_to_index
        self.index_to_lobe = index_to_lobe

        # compute the regions per lobe in a dictionary list
        self._compute_regions_lobe()

    def _compute_regions_in_lobe(self):
        self.lobe_to_regions = collections.defaultdict(list)
        for region, ind in self.region_to_index.items():
            self.lobe_to_regions[self.index_to_lobe[ind]].append(region)

    def get_lobe(self, region_label):
        lobeind = self.region_to_index[region_label]
        return self.index_to_lobe[lobeind]

    def get_regions_for_lobe(self, lobe_label):
        return self.lobe_to_regions[lobe_label]


class Contact:
    def __init__(self, centerX, centerY, centerZ, label):
        self.centerX = centerX  # double
        self.centerY = centerY  # double
        self.centerZ = centerZ  # double

        self.label = label
        self.region1 = 0
        self.region2 = 0
        self.region3 = 0

        self.lobe = None

    def __str__(self):
        return self.label

    def __repr__(self):
        return self.label

    @property
    def coords(self):
        return [self.centerX, self.centerY, self.centerZ]


class Region(object):
    """
    Class for a brain region
    """

    def __init__(self, centerX, centerY, centerZ, label):
        self.centerX = centerX  # double
        self.centerY = centerY  # double
        self.centerZ = centerZ  # double
        self.label = label  # string

        # initialize closest channels up to N=3
        self.chan1 = []
        self.chan2 = []
        self.chan3 = []
        self.contact_lists = [self.chan1, self.chan2, self.chan3]

        # channel count contribution
        self.chancount = 0

    @property
    def summary(self):
        return "BRAIN REGION OBJECT SUMMARY: " \
               "{} with {} contacts".format(self.label, self.chancount)

    def __str__(self):
        return self.label

    def __repr__(self):
        return self.label

    @property
    def contacts(self):
        """
        Property that returns the closest channels
        :return:
        """
        return self.chan1

    def add_nearest_contact(self, contact):
        self.chancount += 1
        self.chan1.append(contact)

    def add_2ndnearest_contact(self, contact):
        self.chancount += .5
        self.chan2.append(contact)

    def add_3rdnearest_contact(self, contact):
        self.chancount += .25
        self.chan3.append(contact)


class Lobe(object):
    """
    Class for a brain lobe
    """

    def __init__(self, name):
        self.numcontacts = 0
        self.numregions = 0
        self.regions = []
        self.contacts = []
        self.label = name

    def __str__(self):
        return self.label

    def __repr__(self):
        return self.label

    @property
    def summary(self):
        return "LOBE OBJECT SUMMARY: " \
               "{} with {} regions and {} contacts".format(self.label,
                                                           self.numregions,
                                                           self.numcontacts)

    def addregion(self, region):
        self.numregions += 1
        self.regions.append(region)

    def addcontact(self, contact):
        self.numcontacts += 1
        self.contacts.append(contact)

    def get_all_contacts(self):
        chanlist1 = []
        chanlist2 = []
        chanlist3 = []

        # go through all regions in this lobe and get their contact lists
        for i in range(len(self.regions)):
            curr_region = self.regions[i]

            # get contacts
            c1list, c2list, c3list = curr_region.contact_lists

            chanlist1.extend(c1list)
            chanlist2.extend(c2list)
            chanlist3.extend(c3list)

        return chanlist1, chanlist2, chanlist3


class Brain():
    """
    Class for a whole brain, which is comprised of many regions and lobes.
    Possibly has implanted channels

    """

    def __init__(self, regionsdict=[], chansdict=[], lobesdict={}):
        self.lobesdict = lobesdict
        self.regionsdict = regionsdict
        self.chansdict = chansdict

        # create lists of objects to hold as part of the brain
        self.regions = self._create_regions_from_dict(regionsdict)
        self.contacts = self._create_contacts_from_dict(chansdict)
        self.lobes = self._create_lobes_from_dict(lobesdict)

        # create a nested list of xyz coords from brain regions
        region_centers = [[reg.centerX, reg.centerY, reg.centerZ]
                          for reg in self.regions]
        # create KD tree to allow fast querying of nearest neighbors in brain space
        self.brain_regions_tree = scipy.spatial.KDTree(region_centers)

        # map contacts to their brain regions
        _brain_regions = self.map_electrodes_to_regions([c.label for c in self.contacts],
                                                        [c.coords for c in self.contacts])

        # compute the inverse of lobe -> contacts mapping.
        self._compute_lobe_to_contact_mapping()
        self._compute_contact_to_lobe_mapping()

    @property
    def summary(self):
        return "BRAIN OBJECT SUMMARY: " \
               "{} Lobes With {} Regions, {} Contacts".format(
                   len(self.lobes),
                   len(self.regions),
                   len(self.contacts)
               )

    def get_lobe_for_contact(self, contactlabel):
        for contact in self.contacts:
            if contactlabel == contact.label:
                return contact.label
        raise RuntimeError("No lobe for this contact?: {}".format(contactlabel))

    def get_contacts_for_lobe(self, lobelabel):
        for lobe in self.lobes:
            if lobelabel == lobe.label:
                return lobe.contacts
        raise RuntimeError("No contacts for this lobe?: {}".format(lobelabel))

    def create_lobe_contact_dict(self):
        lobe_contact_dict = {}
        for lobe in self.lobes:
            lobe_contact_dict[lobe.label] = lobe.contacts
        return lobe_contact_dict

    def compute_hlp(self):
        """
        Function to create dictionary of the hemispheric lobe distribution

        :return: (dict) of hlp lobes: contact_count
        """
        hlp_dict = collections.defaultdict(lambda: 0)

        for region in self.regions:
            # get the computed channel count for this region
            chancount = region.chancount

            # map region to lobe
            for lobe in self.lobes:
                if region.label in lobe.regions:
                    hlp_dict[lobe.label] += chancount

        return hlp_dict

    def compute_nearest_regions(self, elecpos, N=3):
        """
        Function to compute the nearest brain regions for given electrode positions.

        :param elecpos:
        :param region_centers:
        :param N:
        :return:
        """
        closest_centers = []
        c1 = []
        c2 = []

        # use NN-algo to find N nearest nbrs
        dist, inds = self.brain_regions_tree.query(elecpos,
                                                   k=N,
                                                   p=2)
        # print(inds)
        # extract the 3 nearest indices in brain region space
        minind, min2ind, min3ind = inds

        # store closest indices
        closest_centers.append(minind)
        c1.append(min2ind)
        c2.append(min3ind)

        return closest_centers, c1, c2

    def map_electrodes_to_regions(self, chanlabels, chancoords):
        """
        Function to map electrodes with their channel labels and coordinates
        to the nearest brain regions using NN algorithm.

        :param chanlabels: (list) of channel labels
        :param chancoords: (list of (x,y,z)) of their coordinates
        :return:
        """
        for i in range(len(chanlabels)):
            chanlabel = chanlabels[i]
            chancoord = chancoords[i]

            # form nearest neighbor search to get list of closest indices
            closest_centers, c1, c2 = self.compute_nearest_regions(
                chancoord, N=3)

            # add closest centers to each contact
            for i, centre_ind in enumerate(closest_centers):
                # get the region at index
                curr_region = self.regions[centre_ind]
                curr_region.add_nearest_contact(chanlabel)

            for i, centre_ind in enumerate(c1):
                # get the region at index
                curr_region = self.regions[centre_ind]
                curr_region.add_2ndnearest_contact(chanlabel)

            for i, centre_ind in enumerate(c1):
                # get the region at index
                curr_region = self.regions[centre_ind]
                curr_region.add_3rdnearest_contact(chanlabel)

        # return the regions now with appended N close contacts
        return self.regions

    def _compute_lobe_to_contact_mapping(self):
        for lobe in self.lobes:
            # get all the regions in this lobe
            loberegions = lobe.regions

            # loop through each region and get the channels close by
            for region in self.regions:
                # if this is a region label in the lobe, add all those contacts into lobe
                if region.label in loberegions:
                    contacts = region.contacts
                    for contact in contacts:
                        lobe.addcontact(contact)

    def _compute_contact_to_lobe_mapping(self):
        for lobe in self.lobes:
            # find contact that is in lobe already
            for contact in self.contacts:
                if contact in lobe.contacts:
                    contact.lobe = lobe.label

    def _create_regions_from_dict(self, regionsdict):
        allregions = []
        for regionlabel, (x, y, z) in regionsdict.items():
            region = Region(x, y, z, regionlabel)
            allregions.append(region)
        return allregions

    def _create_contacts_from_dict(self, chansdict):
        allcontacts = []
        for contactlabel, (x, y, z) in chansdict.items():
            contact = Contact(x, y, z, contactlabel)
            allcontacts.append(contact)
        return allcontacts

    def _create_lobes_from_dict(self, lobesdict):
        alllobes = []
        for lobelabel, regions_list in lobesdict.items():
            lobe = Lobe(lobelabel)
            for region in regions_list:
                lobe.addregion(region)
            alllobes.append(lobe)
        return alllobes
