import nibabel as nib
import numpy as np

from eegio.format import Contacts
from . import nifti


def mapcontacts_toregs_v2(contacts, label_volume_file):
    # contacts = Contacts(contacts_file)
    label_vol = nib.load(label_volume_file)

    contact_regs = []
    for contact in contacts.names:
        coords = contacts.get_coords(contact)
        region_ind = nifti.point_to_brain_region(
            coords, label_vol, tol=3.0) - 1  # Minus one to account for the shift
        contact_regs.append(region_ind)

    contact_regs = np.array(contact_regs)
    return contact_regs


def mapcontacts_toregs(contacts_file, label_volume_file):
    contacts = Contacts(contacts_file)
    label_vol = nib.load(label_volume_file)

    contact_regs = []
    for contact in contacts.names:
        coords = contacts.get_coords(contact)
        region_ind = nifti.point_to_brain_region(
            coords, label_vol, tol=3.0) - 1  # Minus one to account for the shift
        contact_regs.append(region_ind)

    contact_regs = np.array(contact_regs)
    return contact_regs
