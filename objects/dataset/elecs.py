import collections
import io
import re

import numpy as np
from natsort import natsorted, order_by_index, index_natsorted


class NamedPoints(object):
    def __init__(self, fl):
        data = np.genfromtxt(fl, dtype=None)  # , encoding='utf-8')
        self.xyz = np.array([[l[1], l[2], l[3]] for l in data])
        # self.names = [l[0] for l in data]
        self.names = [l[0].decode('ascii') for l in data]
        self.name_to_xyz = dict(zip(self.names, self.xyz))


class Contacts(object):
    """
    Class wrapper for a set of contacts.

    Allows the user to:
     - load contacts and if available, their corresponding T1 brain xyz coordinates.
     - mask contacts that we don't want to consider
     - set a reference scheme (monopolar, common_average, bipolar, customized)

    Examples
    --------
    >>> from eztrack.eegio.objects.dataset.elecs import Contacts
    >>> contacts_list = ['a1', 'a2', 'a3', 'b1', 'b2', 'b3']
    >>> chan_xyz = [[0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,2,0], [0,3, 0]]
    >>> contacts = Contacts(contacts_list, chan_xyz)
    >>> print("These are the contacts: ", contacts)
    """

    contact_single_regex = re.compile("^([A-Za-z]+[']?)([0-9]+)$")
    contact_pair_regex_1 = re.compile("^([A-Za-z]+[']?)([0-9]+)-([0-9]+)$")
    contact_pair_regex_2 = re.compile(
        "^([A-Za-z]+[']?)([0-9]+)-([A-Za-z]+[']?)([0-9]+)$")

    def __init__(self, contacts_list=[], contacts_xyz=[], require_matching=True):
        self.require_matching = require_matching

        self.chanlabels = list(contacts_list)
        self.xyz = contacts_xyz
        self._initialize_datastructs()

    def __str__(self):
        return ",".join(self.chanlabels)

    def __repr__(self):
        return self.chanlabels

    def _initialize_datastructs(self):
        self.electrodes = collections.defaultdict(list)
        for i, name in enumerate(self.chanlabels):
            match = self.contact_single_regex.match(name)
            if match is None:
                if self.require_matching:
                    raise ValueError("Unexpected contact name %s" % name)
                else:
                    continue

            elec_name, _ = match.groups()
            if elec_name not in self.electrodes:
                self.electrodes[elec_name] = []
            self.electrodes[elec_name].append(i)

    def load_contacts_regions(self, contact_regs):
        self.contact_regs = contact_regs

    def load_from_file(self, filename):
        named_points = NamedPoints(filename)
        self.chanlabels = named_points.names
        self.xyz = named_points.xyz
        self._initialize_datastructs()

    def load_contacts_xyz(self, contacts_xyz):
        self.xyz = contacts_xyz

    def natsort_contacts(self):
        natinds = index_natsorted(self.chanlabels)
        self.chanlabels = np.array(order_by_index(self.chanlabels, natinds))
        return natinds

    def mask_contact_indices(self, mask_inds):
        self.chanlabels = np.array(self.chanlabels)[mask_inds]

    def get_seeg_ngbhrs(self, chsinterest, chanlist):
        nghbrsdict = collections.defaultdict(list)

        # loop through the channels passed in and get their corresponding neighbors
        for idx, ch in enumerate(chsinterest):
            # get the electrode, and the number for each channel
            elecname, num = re.match(
                "^([A-Za-z]+[']?)([0-9]+)$", ch).groups()

            # find the elecname in rest of electrodes
            elecmaxnum = 0
            elec_numstoinds = dict()
            for jdx in range(len(chanlist)):
                _elecname, _num = re.match("^([A-Za-z]+[']?)([0-9]+)$", chanlist[jdx]).groups()
                if elecname == _elecname:
                    elecmaxnum = max(elecmaxnum, int(_num))
                    elec_numstoinds[_num] = jdx

            # find keys with number above and below number
            elecnumkeys = natsorted(elec_numstoinds.keys())

            elecnumkeys = np.arange(1, elecmaxnum).astype(str).tolist()
            if num not in elecnumkeys:
                continue

            currnumind = elecnumkeys.index(num)
            lowerind = max(0, currnumind-2)
            upperind = min(int(elecnumkeys[-1]), currnumind+2)

            # print(elecnumkeys)
            # print(num)
            # print(currnumind)
            # print(elecnumkeys[currnumind-2:currnumind+2])
            nghbrsdict[ch].extend([elecname + x for x in elecnumkeys[lowerind:upperind]])

            # if num == 1:
            #     nghbrsdict[ch].extend([elecname+"2", elecname+"3"])
            # elif num == 2:
        return nghbrsdict

    def set_bipolar(self, chanlabels=[]):
        self.chanlabels_bipolar = []
        self._bipolar_inds = []

        remaining_labels = np.array([])
        # apply to channel labels, if none are passed in
        if len(chanlabels) == 0:
            chanlabels = self.chanlabels
        else:
            # get the remaining labels in the channels
            remaining_labels = np.array(
                [ch for ch in self.chanlabels if ch not in chanlabels])
        n = len(chanlabels)

        for inds in zip(np.r_[:n - 1], np.r_[1:n]):
            # get the names for all the channels
            names = [chanlabels[ind] for ind in inds]

            # get the electrode, and the number for each channel
            elec0, num0 = re.match(
                "^([A-Za-z]+[']?)([0-9]+)$", names[0]).groups()
            elec1, num1 = re.match(
                "^([A-Za-z]+[']?)([0-9]+)$", names[1]).groups()

            # if electrode name matches, and the number are off by 1, then apply bipolar
            if elec0 == elec1 and abs(int(num0) - int(num1)) == 1:
                name = "%s%s-%s" % (elec0, num1, num0)
                self.chanlabels_bipolar.append(name)
                self._bipolar_inds.append([inds[1], inds[0]])

        # convert to numpy arrays and convert the raw data
        self._bipolar_inds = np.array(self._bipolar_inds, dtype=int)
        self.chanlabels = self.chanlabels_bipolar

        if remaining_labels.size == 0:
            return self._bipolar_inds
        else:
            return self._bipolar_inds, remaining_labels

    def set_localreference(self, chanlabels=[], chantypes=[]):
        """
        Only applies for SEEG and Strip channels.

        TODO: Can run computation for grids

        http://www.jneurosci.org/content/31/9/3400
        :param chanlabels:
        :return:
        """
        remaining_labels = np.array([])
        # apply to channel labels, if none are passed in
        if len(chanlabels) == 0:
            chanlabels = self.chanlabels
        else:
            # get the remaining labels in the channels
            remaining_labels = np.array(
                [ch for ch in self.chanlabels if ch not in chanlabels])
        n = len(chanlabels)

        ''' ASSUME ALL SEEG OR STRIP FOR NOW '''
        if chantypes == []:
            chantypes = ['seeg' for i in range(n)]

        if any(chantype not in ['seeg', 'grid', 'strip'] for chantype in chantypes):
            raise ValueError("Channel types can only be of seeg, grid, or strip. "
                             "Make sure you pass in valid channel types! You passed in {}".format(chantypes))

        # # first naturally sort contacts
        # natinds = index_natsorted(self.chanlabels)
        # sortedchanlabels = self.chanlabels.copy()
        # sortedchanlabels = sortedchanlabels[natinds]

        # # get end indices on the electrode that don't have a local reference
        # endinds = []
        # for elec in self.electrodes.keys():
        #     endinds.extend([self.electrodes[elec][0], self.electrodes[elec][-1]])

        # create a dictionary to store all key/values of channels/nbrs
        localreferenceinds = collections.defaultdict(list)

        # get all neighboring channels
        for i, (electrode, inds) in enumerate(self.electrodes.items()):
            for ind in inds:
                # get for this index the channel type
                chantype = chantypes[ind]
                chanlabel = self.chanlabels[ind]

                if chantype == 'grid':
                    # get all grid channels
                    gridlayout = [self.chanlabels[g_ind] for g_ind in inds]
                    # run 2d neighbors
                    nbrs_chans, nbrs_inds = self._get2d_neighbors(
                        gridlayout, chanlabel)

                else:
                    # get all channels for this electrode
                    electrodechans = [self.chanlabels[elec_ind]
                                      for elec_ind in inds]
                    # run 1d neighbors
                    nbrs_chans, nbrs_inds = self._get1d_neighbors(
                        electrodechans, chanlabel)

                # create dictionary of neighbor channels
                localreferenceinds[chanlabel] = nbrs_chans

        # loop through combinations of channels
        # for inds in zip(np.r_[:n - 2], np.r_[1:n - 1], np.r_[2:n]):
        #     # get the names for all the channels
        #     names = [sortedchanlabels[ind] for ind in inds]
        #
        #     # get the electrode, and the number for each channel
        #     elec0, num0 = re.match("^([A-Za-z]+[']?)([0-9]+)$", names[0]).groups()
        #     elec1, num1 = re.match("^([A-Za-z]+[']?)([0-9]+)$", names[1]).groups()
        #     elec2, num2 = re.match("^([A-Za-z]+[']?)([0-9]+)$", names[2]).groups()
        #
        #     # if electrode name matches, and the number are off by 1, then apply bipolar
        #     if elec0 == elec1 and elec1 == elec2 and abs(int(num0) - int(num1)) == 1 and abs(
        #             int(num1) - int(num2)) == 1:
        #         localreferenceinds[inds[1]] = [inds[0], inds[2]]

        # get indices for all the local referenced channels
        self.localreferencedict = localreferenceinds

        # compute leftover channels
        leftoverinds = [ind for ind, ch in enumerate(
            chanlabels) if ch not in localreferenceinds.keys()]
        self.leftoverchanlabels = self.chanlabels[leftoverinds]

        if remaining_labels.size == 0:
            return self.localreferencedict
        else:
            return self.localreferencedict, self.leftoverchanlabels

    def _get1d_neighbors(self, electrodechans, chanlabel):
        # naturally sort the gridlayout
        natinds = index_natsorted(electrodechans)
        sortedelec = electrodechans[natinds]

        # get the channel index - ijth index
        chanindex = sortedelec.index(chanlabel)

        # get the neighboring indices and channels
        nbrinds = [chanindex + 1, chanindex - 1]
        nbrchans = [chanlabel[ind] for ind in nbrinds]

        return nbrchans, nbrinds

    def _get2d_neighbors(self, gridlayout, chanlabel):
        """
        Helper function to retrun the 2D neighbors of a certain
        channel label based on.

        TODO: ensure the grid layout logic is correct. Assumes a certain grid structure.

        :param gridlayout:
        :param chanlabel:
        :return:
        """

        def convert_index_to_grid(chanindex, numcols, numrows):
            """
            Helper function with indices from 0-7, 0-31, etc.
            Index starts at 0.

            :param chanindex:
            :param numcols:
            :param numrows:
            :return:
            """
            # get column and row of channel index
            chanrow = np.ceil(chanindex / numcols)
            chancol = numcols - ((chanrow * numcols) - chanindex)

            if chanrow > numrows:
                raise RuntimeError("How is row of this channel greater then number of rows set!"
                                   "Error in function, or logic, or data passed in!")

            return chanrow, chancol

        def convert_grid_to_index(i, j, numcols, numrows):
            """
            Helper function with indices from 1-32, 1-64, etc.
            Index starts at 1.

            :param i:
            :param j:
            :param numcols:
            :param numrows:
            :return:
            """
            if i > numrows:
                raise RuntimeError("How is row of this channel greater then number of rows set!"
                                   "Error in function, or logic, or data passed in!")

            chanindex = 0
            chanindex += (i - 1) * numcols
            chanindex += j
            return chanindex

        # naturally sort the gridlayout
        natinds = index_natsorted(gridlayout)
        sortedgrid = gridlayout[natinds]

        # determine number of rows/cols in the grid
        numcols = 8
        numrows = len(sortedgrid) // numcols

        # get the channel index - ijth index
        chanindex = sortedgrid.index(chanlabel)
        chanrow, chancol = convert_index_to_grid(chanindex, numcols, numrows)

        # get the neighbor indices
        twoDnbrs_inds = [
            convert_grid_to_index(chanrow + 1, chancol, numcols, numrows),
            convert_grid_to_index(chanrow - 1, chancol, numcols, numrows),
            convert_grid_to_index(chanrow, chancol + 1, numcols, numrows),
            convert_grid_to_index(chanrow, chancol - 1, numcols, numrows),
        ]
        # get the channels of neighbors
        twoDnbrs_chans = [sortedgrid[ind] for ind in twoDnbrs_inds]
        return twoDnbrs_chans, twoDnbrs_inds

    def get_elec(self, name):
        match = self.contact_single_regex.match(name)
        if match is None:
            return None
        return match.groups()[0]

    def get_coords(self, name):
        """Get the coordinates of a specified contact or contact pair. Allowed formats are:

        A1            : Single contact.
        A1-2 or A1-A2 : Contact pair. The indices must be adjacent.


        Examples:

        >>> np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})

        >>> contacts = Contacts(io.BytesIO("A1 0.0 0.0 1.0\\nA2 0.0 0.0 2.0".encode()))
        >>> contacts.get_coords("A1")
        array([0.0, 0.0, 1.0])

        >>> contacts.get_coords("A1-2")
        array([0.0, 0.0, 1.5])

        >>> contacts.get_coords("A2-A1")
        array([0.0, 0.0, 1.5])
        """

        match = self.contact_single_regex.match(name)
        if match is not None:
            return self.name_to_xyz[name]

        match = self.contact_pair_regex_1.match(name)
        if match is not None:
            assert abs(int(match.group(2)) - int(match.group(3))) == 1
            contact1 = match.group(1) + match.group(2)
            contact2 = match.group(1) + match.group(3)
            return (self.name_to_xyz[contact1] +
                    self.name_to_xyz[contact2]) / 2.

        match = self.contact_pair_regex_2.match(name)
        if match is not None:
            assert match.group(1) == match.group(3)
            assert abs(int(match.group(2)) - int(match.group(4))) == 1
            contact1 = match.group(1) + match.group(2)
            contact2 = match.group(3) + match.group(4)
            return (self.name_to_xyz[contact1] +
                    self.name_to_xyz[contact2]) / 2.

        raise ValueError(
            "Given name '%s' does not follow any expected pattern." %
            name)
