# -*- coding: utf-8 -*-

import re
import zipfile

import nibabel as nib
import numpy as np
import pandas as pd
import datetime

from eegio.base.objects import Contacts
from . import nifti


def add_same_occurence_index(df, column):
    df['_%s_repeated' % column] = False
    df['_%s_index' % column] = 1

    for key in pd.unique(df[column]):
        if pd.isna(key):
            continue

        subdf = df[df[column] == key]
        if len(subdf) > 1:
            for i, (index, row) in enumerate(subdf.iterrows()):
                df.loc[index, '_%s_repeated' % column] = True
                df.loc[index, '_%s_index' % column] = i + 1


def get_ez_from_regions(xlsx_file, region_names):
    """Return list of indices of EZ regions given in the patient spreadsheet"""

    LH_NAMES_IND = 9
    LH_EZ_IND = 10
    RH_NAMES_IND = 12
    RH_EZ_IND = 13

    df = pd.read_excel(xlsx_file, sheet_name="EZ hypothesis and EI", header=1)

    ez_names = []
    for names_ind, ez_ind in [
            (LH_NAMES_IND, LH_EZ_IND), (RH_NAMES_IND, RH_EZ_IND)]:
        names_col = df.iloc[:, names_ind]
        mask = names_col.notnull()
        names = names_col[mask]
        ez_mask = df.iloc[:, ez_ind][mask].astype(str) == 'YES'
        ez_names.extend(names[ez_mask])

    return [region_names.index(name) for name in ez_names]


def get_ez_from_contacts(xlsx_file, contacts_file, label_volume_file):
    """Return list of indices of EZ regions given by the EZ contacts in the patient spreadsheet"""

    CONTACTS_IND = 6
    EZ_IND = 7

    df = pd.read_excel(xlsx_file, sheet_name="EZ hypothesis and EI", header=1)

    ez_contacts = []
    contacts_col = df.iloc[:, CONTACTS_IND]
    mask = contacts_col.notnull()
    contacts_names = contacts_col[mask]
    ez_mask = df.iloc[:, EZ_IND][mask] == 'YES'
    ez_contacts.extend(contacts_names[ez_mask])

    contacts = Contacts(contacts_file)
    label_vol = nib.load(label_volume_file)

    ez_inds = []
    for contact in ez_contacts:
        coords = contacts.get_coords(contact)
        region_ind = nifti.point_to_brain_region(
            coords, label_vol, tol=3.0) - 1  # Minus one to account for the shift
        if region_ind != -1:
            ez_inds.append(region_ind)

    return ez_inds


def save_ez_hypothesis(xlsx_file, tvb_zipfile,
                       contacts_file, label_volume_file, output_file):
    """Extract the EZ hypothesis from the xlsx file and save it to plain text file"""

    with zipfile.ZipFile(tvb_zipfile) as zf:
        with zf.open("centres.txt") as fl:
            region_names = list(np.genfromtxt(fl, usecols=(0,), dtype=str))

    nreg = len(region_names)

    ez_inds_from_regions = get_ez_from_regions(xlsx_file, region_names)
    ez_inds_from_contacts = get_ez_from_contacts(
        xlsx_file, contacts_file, label_volume_file)
    ez_inds = list(set(ez_inds_from_regions + ez_inds_from_contacts))

    ez_hyp = np.zeros(nreg, dtype=int)
    ez_hyp[ez_inds] = 1

    np.savetxt(output_file, ez_hyp, fmt='%i')


def expand_channels(ch_list):
    ch_list = [a.replace("’", "'").replace(
        "\n", "").replace(" ", "") for a in ch_list]

    new_list = []
    for string in ch_list:
        if string == 'nan':
            continue

        if not string.strip():
            continue

        # A'1,2,5,7
        match = re.match("^([A-Za-z]+[']*)([0-9,]*)([A-Za-z]*)$", string)
        if match:
            name, fst_idx, last_idx = match.groups()
            numbers = fst_idx.split(',')
            new_list.extend([name + str(char)
                             for char in numbers if char != ","])
            continue

        # A'1
        match = re.match("^([A-Za-z]+[']*)([0-9]+)$", string)
        if match:
            new_list.append(string)
            continue

        # A'1-10
        match = re.match("^([A-Za-z]+[']*)([0-9]+)-([0-9]+)$", string)
        if match:
            name, fst_idx, last_idx = match.groups()
            new_list.extend([name + str(i)
                             for i in range(int(fst_idx), int(last_idx) + 1)])
            continue

        # A'1-A10
        match = re.match(
            "^([A-Za-z]+[']*)([0-9]+)-([A-Za-z]+[']*)([0-9]+)$",
            string)
        if match:
            name1, fst_idx, name2, last_idx = match.groups()
            if name1 == name2:
                new_list.extend([name1 + str(i)
                                 for i in range(int(fst_idx), int(last_idx) + 1)])
                continue

        # A'1,B'1,
        match = re.match("^([A-Za-z]+[']*)([0-9,])([A-Za-z]*)$", string)
        if match:
            name, fst_idx, last_idx = match.groups()
            numbers = fst_idx.split(',')
            new_list.extend([name + str(char)
                             for char in numbers if char != ","])
            continue

        match = string.split(',')
        if match:
            new_list.extend([ch for ch in match])
            continue
        #             print("Printing: ", match)
        #
        # ops = re.findall(r"\d+", string)  # r"\d+" searches for digits of variables length
        # if ops:
        #     prefix = re.findall(r"\D+", string)[0]  # r"\D+" complement set of "\d+"
        #     new_list.extend([prefix + str(i) for i in list(range(int(ops[0]), int(ops[1]), 1))])
        #     continue

        print("expand_channels: Cannot parse this: %s" % string)

    return new_list


def get_sec(time):
    if isinstance(time, float):
        # Already in seconds
        return time
    elif isinstance(time, datetime.time):
        return datetime.timedelta(hours=time.hour,
                                  minutes=time.minute,
                                  seconds=time.second,
                                  microseconds=time.microsecond).total_seconds()
    elif isinstance(time, str):
        h, m, s = time.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    else:
        raise ValueError("Unexpected time type: %s" % type(time))


class FormatChannelString():
    def __init__(self, channelstring):
        self.channelstring = channelstring

        elecs, inds = self._unique_electrodes()
        self.elecs = elecs

    def _unique_electrodes(self, chanlabels):
        # determine all unique electrodes
        chanlabels = [self._remove_digits(chan) for chan in chanlabels]
        elecs, inds = np.unique(chanlabels, return_index=True)

        # return non-sorted version
        elecs = [chanlabels[index] for index in sorted(inds)]

        return elecs, inds

    # first aggregate by all electrodes
    def _remove_digits(self, string):
        result = ''.join([i for i in string if not i.isdigit()])
        return result

    def help_aggregate_contacts(self, contacts):
        elecs, inds = self._unique_electrodes(contacts)

        new_aggr_contacts = []

        for idx, elec in enumerate(elecs):
            deep_cont = []
            medial_cont = []
            lateral_cont = []

            # get the indices for this electrode
            elec_inds = [ind for ind, chan in enumerate(contacts)
                         if self._remove_digits(chan) == elec]
            # loop over all this electrode's indices
            for elec_ind in elec_inds:
                chanlabel = contacts[elec_ind]
                contact_number = int(chanlabel.replace(elec, ""))

                # assign to their corresponding lists
                if contact_number in range(1, 5):
                    deep_cont.append(elec_ind)
                elif contact_number in range(5, 9):
                    medial_cont.append(elec_ind)
                elif contact_number in range(9, 13):
                    lateral_cont.append(elec_ind)

            # create final list to hold all contacts
            aggregated_conts = [deep_cont, medial_cont, lateral_cont]
            # loop through all aggregated contacts
            for cdx, conts in enumerate(aggregated_conts):
                if conts:
                    # create aggregated matrix and labels
                    if cdx == 0:
                        # create new labels for these contacts
                        new_aggr_contacts.append(elec + '1-4')
                    elif cdx == 1:
                        new_aggr_contacts.append(elec + '5-8')
                    elif cdx == 2:
                        new_aggr_contacts.append(elec + '9-12')
        return new_aggr_contacts

    # def aggregate_contacts(self):
    #     # initialize new matrix to store
    #     new_mat = np.zeros((len(self.elecs) * 3, self.fragmat.shape[1]))
    #     new_aggr_contacts = []
    #
    #     new_mat = []
    #
    #     # loop through each electrode
    #     for idx, elec in enumerate(self.elecs):
    #         deep_cont = []
    #         medial_cont = []
    #         lateral_cont = []
    #
    #         # get the indices for this electrode
    #         elec_inds = [ind for ind, chan in enumerate(self.chanlabels)
    #                      if self._remove_digits(chan) == elec]
    #         # loop over all this electrode's indices
    #         for elec_ind in elec_inds:
    #             chanlabel = self.chanlabels[elec_ind]
    #             contact_number = int(chanlabel.replace(elec, ""))
    #
    #             # assign to their corresponding lists
    #             if contact_number in range(1, 5):
    #                 deep_cont.append(elec_ind)
    #             elif contact_number in range(5, 9):
    #                 medial_cont.append(elec_ind)
    #             elif contact_number in range(9, 13):
    #                 lateral_cont.append(elec_ind)
    #
    #         aggregated_conts = [deep_cont, medial_cont, lateral_cont]
    #
    #         for cdx, conts in enumerate(aggregated_conts):
    #
    #             if conts:
    #                 frag = np.mean(self.fragmat[conts, :], axis=0)
    #
    #                 # create aggregated matrix and labels
    #                 new_mat.append(frag)
    #                 if cdx == 0:
    #                     # create new labels for these contacts
    #                     new_aggr_contacts.append(elec + '1-4')
    #                 elif cdx == 1:
    #                     new_aggr_contacts.append(elec + '5-8')
    #                 elif cdx == 2:
    #                     new_aggr_contacts.append(elec + '9-12')
    #
    #         # print(deep_cont)
    #         # print(medial_cont)
    #         # print(lateral_cont)
    #         # # compute the fragility at these different levels of contacts
    #         # deep_frag = np.mean(self.fragmat[deep_cont,:], axis=0)
    #         # medial_frag = np.mean(self.fragmat[medial_cont,:], axis=0)
    #         # lateral_frag = np.mean(self.fragmat[lateral_cont,:], axis=0)
    #
    #         # pointer = idx*3
    #         # new_mat[pointer,:] = deep_frag
    #         # new_mat[pointer+1,:] = medial_frag
    #         # new_mat[pointer+2,:] = lateral_frag
    #
    #         # # create new labels for these contacts
    #         # new_aggr_contacts.append(elec+'1-4')
    #         # new_aggr_contacts.append(elec+'5-8')
    #         # new_aggr_contacts.append(elec+'9-12')
    #
    #     self.aggregated_labels = np.array(new_aggr_contacts)
    #     self.aggregated_mat = np.array(new_mat)
    #     return new_mat
    #
    # def aggregate_electrodes(self):
    #     elec_hash = dict()
    #     new_mat = np.zeros((len(self.elecs), self.fragmat.shape[1]))
    #     for idx, elec in enumerate(self.elecs):
    #         # get the indices for this electrode
    #         elec_inds = [ind for ind, chan in enumerate(self.chanlabels)
    #                      if self._remove_digits(chan) == elec]
    #         elec_hash[elec] = elec_inds
    #
    #         new_mat[idx, :] = np.mean(self.fragmat[elec_inds, :], axis=0)
    #
    #     self.aggregated_labels = np.array(self.elecs)
    #     self.aggregated_mat = np.array(new_mat)
    #     return new_mat
