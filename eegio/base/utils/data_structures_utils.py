# Data structure manipulations and conversions
import contextlib
import io
import json
import ntpath
import os
import re
from collections import OrderedDict
from copy import deepcopy
from datetime import date, datetime

import numpy as np
import scipy.io as sio
from natsort import index_natsorted


# class NamedPoints(object):
# #     def __init__(self, fl):
# #         data = np.genfromtxt(fl, dtype=None)  # , encoding='utf-8')
# #         self.xyz = np.array([[l[1], l[2], l[3]] for l in data])
# #         # self.names = [l[0] for l in data]
# #         self.names = [l[0].decode('ascii') for l in data]
# #         self.name_to_xyz = dict(zip(self.names, self.xyz))
# #
# #     def load_contacts_regions(self, contact_regs):
# #         self.contact_regs = contact_regs
# #
# #     def load_from_file(self, filename):
# #         named_points = NamedPoints(filename)
# #         self.chanlabels = named_points.names
# #         self.xyz = named_points.xyz
# #         self._initialize_datastructs()


def _get1d_neighbors(electrodechans, chanlabel):
    """

    Parameters
    ----------
    electrodechans :
    chanlabel :

    Returns
    -------

    """
    # naturally sort the gridlayout
    natinds = index_natsorted(electrodechans)
    sortedelec = electrodechans[natinds]

    # get the channel index - ijth index
    chanindex = sortedelec.index(chanlabel)

    # get the neighboring indices and channels
    nbrinds = [chanindex + 1, chanindex - 1]
    nbrchans = [chanlabel[ind] for ind in nbrinds]

    return nbrchans, nbrinds


def _get2d_neighbors(gridlayout, chanlabel):
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
            raise RuntimeError(
                "How is row of this channel greater then number of rows set!"
                "Error in function, or logic, or data passed in!"
            )

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
            raise RuntimeError(
                "How is row of this channel greater then number of rows set!"
                "Error in function, or logic, or data passed in!"
            )

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


def load_szinds(onsetind, offsetind, timepoints):
    timepoints = np.array(timepoints)
    # get the actual indices that occur within time windows
    if onsetind is not None:
        onsetwin = findtimewins(onsetind, timepoints)
    else:
        onsetwin = None
    if offsetind is not None:
        offsetwin = findtimewins(offsetind, timepoints)
    else:
        offsetwin = None
    return onsetwin, offsetwin


def findtimewins(times, timepoints):
    indices = []
    for time in ensure_list(times):
        if time == 0:
            indices.append(time)
        else:
            idx = (time >= timepoints[:, 0]) * (time <= timepoints[:, 1])
            timeind = np.where(idx)[0]
            if len(timeind) > 0:
                indices.append(timeind[0])
            else:
                indices.append(np.nan)
    return indices


def compute_samplepoints(winsamps, stepsamps, numtimepoints):
    # Creates a [n,2] array that holds the sample range of each window that
    # is used to index the raw data for a sliding window analysis
    samplestarts = np.arange(0, numtimepoints - winsamps + 1.0, stepsamps).astype(int)
    sampleends = np.arange(winsamps, numtimepoints + 1, stepsamps).astype(int)

    # print(len(sampleends), len(samplestarts))
    samplepoints = np.append(
        samplestarts[:, np.newaxis], sampleends[:, np.newaxis], axis=1
    )
    return samplepoints


def writejsonfile(metadata, metafilename, overwrite=False):
    if os.path.exists(metafilename) and overwrite is False:
        raise OSError(
            "Destination for meta json file exists! Please use option"
            " overwrite=True to force overwriting."
        )
    with io.open(metafilename, "w", encoding="utf8") as outfile:
        str_ = json.dumps(
            metadata,
            indent=4,
            sort_keys=True,
            cls=NumpyEncoder,
            separators=(",", ": "),
            ensure_ascii=False,
        )
        outfile.write(str_)


def loadjsonfile(metafilename):
    if not metafilename.endswith(".json"):
        metafilename += ".json"

    # encoding = "ascii", errors = "surrogateescape"
    try:
        with open(metafilename, mode="r", encoding="utf8", errors="ignore") as f:
            metadata = json.load(f)
        metadata = json.loads(metadata)
    except Exception as e:
        # print(e)
        with io.open(metafilename, errors="ignore", encoding="utf8", mode="r") as fp:
            json_str = fp.read()
        # print(json_str)
        try:
            metadata = json.loads(json_str)
        except:
            print(json_str)
        # with open(metafilename, mode='r', encoding='utf8', errors="ignore") as f:
        #     metadata = json.load(f)
        # metadata = json.loads(metadata)
    return metadata


def walk_up_folder(path, depth=1):
    _cur_depth = 1
    while _cur_depth < depth:
        path = os.path.dirname(path)
        _cur_depth += 1
    return path


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def compute_timepoints(numsignals, winsize, stepsize, samplerate):
    timepoints_ms = numsignals / samplerate

    # create array of indices of window start and end times
    timestarts = np.arange(0, timepoints_ms - winsize + 1, stepsize)
    timeends = np.arange(winsize - 1, timepoints_ms, stepsize)
    # create the timepoints array for entire data array
    timepoints = np.append(timestarts[:, np.newaxis], timeends[:, np.newaxis], axis=1)

    return timepoints


def merge_metadata(metadata1, metadata2, overwrite=False):
    for key in metadata1.keys():
        if overwrite is False:
            if key not in metadata2.keys():
                metadata2[key] = metadata1[key]
        else:
            metadata2[key] = metadata1[key]
    return metadata2


class MatReader:
    """
    Object to read mat files into a nested dictionary if need be.
    Helps keep strucutre from matlab similar to what is used in python.
    """

    def __init__(self, filename=None):
        self.filename = filename

    def loadmat(self, filename):
        """
        this function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects
        """
        data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
        return self._check_keys(data)

    def _check_keys(self, dict):
        """
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in dict:
            if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
                dict[key] = self._todict(dict[key])
        return dict

    def _todict(self, matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, sio.matlab.mio5_params.mat_struct):
                dict[strg] = self._todict(elem)
            elif isinstance(elem, np.ndarray):
                dict[strg] = self._tolist(elem)
            else:
                dict[strg] = elem
        return dict

    def _tolist(self, ndarray):
        """
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, sio.matlab.mio5_params.mat_struct):
                elem_list.append(self._todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(self._tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    def convertMatToJSON(self, matData, fileName):
        jsonData = {}

        for key in matData.keys():
            if (type(matData[key])) is np.ndarray:
                serializedData = pickle.dumps(
                    matData[key], protocol=0
                )  # protocol 0 is printable ASCII
                jsonData[key] = serializedData
            else:
                jsonData[key] = matData[key]

        with contextlib.closing(bz2.BZ2File(fileName, "wb")) as f:
            json.dump(jsonData, f)


def writematfile(matfilename, **kwargs):
    """
    Function used to write to a mat file.

    We will need the matrix CxT and contact regs of Cx1
    and also the regmap vector that will be Nx1.

    or

    matrix of RxT (averaged based on contact_regs) and the Nx1
    reg map vector that will be used to map each of the vertices
    to a region in R.

    Using these, matlab can easily assign the triangles that
    belong to regions for each channel/region and color with
    that color according to the colormap defined by the function.
    """
    if not matfilename.endswith(".mat"):
        matfilename += ".mat"
    sio.savemat(matfilename, kwargs)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  # This is the fix
            return obj.tolist()
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


def vector2scalar(x):
    if not (isinstance(x, np.ndarray)):
        return x
    else:
        y = np.squeeze(x)
    if all(y.squeeze() == y[0]):
        return y[0]
    else:
        return reg_dict(x)


def list_of_strings_to_string(lstr, sep=","):
    result_str = lstr[0]
    for s in lstr[1:]:
        result_str += sep + s
    return result_str


def dict_str(d):
    s = "{"
    for key, value in d.items():
        s += "\n" + key + ": " + str(value)
    s += "}"
    return s


def isequal_string(a, b, case_sensitive=False):
    if case_sensitive:
        return a == b
    else:
        try:
            return a.lower() == b.lower()
        except AttributeError:
            logger.warning("Case sensitive comparison!")
            return a == b


def split_string_text_numbers(ls):
    items = []
    for s in ensure_list(ls):
        match = re.findall(r"(\d+|\D+)", s)
        if match:
            items.append(tuple(match[:2]))
    return items


def construct_import_path(path, package="tvb_epilepsy"):
    path = path.split(".py")[0]
    start = path.find(package)
    return path[start:].replace("/", ".")


def formal_repr(instance, attr_dict, sort_dict_flag=False):
    """ A formal string representation for an object.
    :param attr_dict: dictionary attribute_name: attribute_value
    :param instance:  Instance to read class name from it
    """
    class_name = instance.__class__.__name__
    formal = class_name + "{"
    if sort_dict_flag:
        attr_dict = sort_dict(attr_dict)
    for key, val in attr_dict.items():
        if isinstance(val, dict):
            formal += "\n" + key + "=["
            for key2, val2 in val.items():
                formal += "\n" + str(key2) + " = " + str(val2)
            formal += "]"
        else:
            formal += "\n" + str(key) + " = " + str(val)
    return formal + "}"


def obj_to_dict(obj):
    """
    :param obj: Python object to introspect
    :return: dictionary after recursively taking obj fields and their values
    """
    if obj is None:
        return obj
    if isinstance(obj, (str, int, float)):
        return obj
    if isinstance(obj, (np.float32,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, list):
        ret = []
        for val in obj:
            ret.append(obj_to_dict(val))
        return ret
    ret = {}
    for key in obj.__dict__:
        val = getattr(obj, key, None)
        ret[key] = obj_to_dict(val)
    return ret


def sort_dict(d):
    return OrderedDict(sorted(d.items(), key=lambda t: t[0]))


def dicts_of_lists(dictionary, n=1):
    for key, value in dictionary.items():
        dictionary[key] = ensure_list(dictionary[key])
        if len(dictionary[key]) == 1 and n > 1:
            dictionary[key] = dictionary[key] * n
    return dictionary


def iterable_to_dict(obj):
    d = OrderedDict()
    for ind, value in enumerate(obj):
        d["%02d" % ind] = value
    return d


def dict_to_list_or_tuple(dictionary, output_obj="list"):
    dictionary = sort_dict(dictionary)
    output = dictionary.values()
    if output_obj == "tuple":
        output = tuple(output)
    return output


def list_of_dicts_to_dicts_of_ndarrays(lst, shape=None):
    d = dict(zip(lst[0], zip(*list([d.values() for d in lst]))))
    if isinstance(shape, tuple):
        for key, val in d.items():
            d[key] = np.reshape(np.stack(d[key]), shape)
    else:
        for key, val in d.items():
            d[key] = np.squeeze(np.stack(d[key]))
    return d


def arrays_of_dicts_to_dicts_of_ndarrays(arr):
    lst = arr.flatten().tolist()
    d = list_of_dicts_to_dicts_of_ndarrays(lst)
    for key, val in d.items():
        d[key] = np.reshape(d[key], arr.shape)
    return d


def dicts_of_lists_to_lists_of_dicts(dictionary):
    return [dict(zip(dictionary, t)) for t in zip(*dictionary.values())]


def ensure_string(arg):
    if not (isinstance(arg, str)):
        if arg is None:
            return ""
        else:
            return ensure_list(arg)[0]
    else:
        return arg


def ensure_list(arg):
    if not (isinstance(arg, list)):
        try:  # if iterable
            if isinstance(arg, (str, dict)):
                arg = [arg]
            else:
                arg = list(arg)
        except BaseException:  # if not iterable
            arg = [arg]
    return arg


def ensure_string(arg):
    if not (isinstance(arg, str)):
        if arg is None:
            return ""
        else:
            return ensure_list(arg)[0]
    else:
        return arg


def set_list_item_by_reference_safely(ind, item, lst):
    while ind >= len(lst):
        lst.append(None)
    lst.__setitem__(ind, item)


def get_list_or_tuple_item_safely(obj, key):
    try:
        return obj[int(key)]
    except BaseException:
        return None


def linear_index_to_coordinate_tuples(linear_index, shape):
    if len(linear_index) > 0:
        coordinates_tuple = np.unravel_index(linear_index, shape)
        return zip(*[ca.flatten().tolist() for ca in coordinates_tuple])
    else:
        return []


def labels_to_inds(labels, lbls):
    idx = []
    lbls = ensure_list(lbls)
    for i, label in enumerate(labels):
        if label in lbls:
            idx.append(i)
    return np.unique(idx)


def generate_region_labels(n_regions, labels=[], str=". ", numbering=True):
    if len(labels) == n_regions:
        if numbering:
            return np.array(
                [
                    str.join(["%d", "%s"]) % tuple(l)
                    for l in zip(range(n_regions), labels)
                ]
            )
        else:
            return labels
    else:
        return np.array(["%d" % l for l in range(n_regions)])


# This function is meant to confirm that two objects assumingly of the
# same type are equal, i.e., identical
def assert_equal_objects(obj1, obj2, attributes_dict=None, logger=None):
    def print_not_equal_message(attr, field1, field2, logger):
        # logger.error("\n\nValueError: Original and read object field "+ attr + " not equal!")
        # raise_value_error("\n\nOriginal and read object field " + attr + " not equal!")
        logger.warning(
            "Original and read object field "
            + attr
            + " not equal!"
            + "\nOriginal field:\n"
            + str(field1)
            + "\nRead object field:\n"
            + str(field2),
            logger,
        )

    if isinstance(obj1, dict):

        def get_field1(obj, key):
            return obj[key]

        if not (isinstance(attributes_dict, dict)):
            attributes_dict = dict()
            for key in obj1.keys():
                attributes_dict.update({key: key})
    elif isinstance(obj1, (list, tuple)):

        def get_field1(obj, key):
            return get_list_or_tuple_item_safely(obj, key)

        indices = range(len(obj1))
        attributes_dict = dict(zip([str(ind) for ind in indices], indices))
    else:

        def get_field1(obj, attribute):
            return getattr(obj, attribute)

        if not (isinstance(attributes_dict, dict)):
            attributes_dict = dict()
            for key in obj1.__dict__.keys():
                attributes_dict.update({key: key})
    if isinstance(obj2, dict):

        def get_field2(obj, key):
            return obj.get(key, None)

    elif isinstance(obj2, (list, tuple)):

        def get_field2(obj, key):
            return get_list_or_tuple_item_safely(obj, key)

    else:

        def get_field2(obj, attribute):
            return getattr(obj, attribute, None)

    equal = True
    for attribute in attributes_dict:
        # print attributes_dict[attribute]
        field1 = get_field1(obj1, attributes_dict[attribute])
        field2 = get_field2(obj2, attributes_dict[attribute])
        try:
            # TODO: a better hack for the stupid case of an ndarray of a string, such as model.zmode or pmode
            # For non numeric types
            if (
                isinstance(field1, str)
                or isinstance(field1, list)
                or isinstance(field1, dict)
                or (isinstance(field1, np.ndarray) and field1.dtype.kind in "OSU")
            ):
                if np.any(field1 != field2):
                    print_not_equal_message(
                        attributes_dict[attribute], field1, field2, logger
                    )
                    equal = False
            # For numeric numpy arrays:
            elif isinstance(field1, np.ndarray) and not field1.dtype.kind in "OSU":
                # TODO: handle better accuracy differences, empty matrices and
                # complex numbers...
                if field1.shape != field2.shape:
                    print_not_equal_message(
                        attributes_dict[attribute], field1, field2, logger
                    )
                    equal = False
                elif np.any(np.float32(field1) - np.float32(field2) > 0):
                    print_not_equal_message(
                        attributes_dict[attribute], field1, field2, logger
                    )
                    equal = False
            # For numeric scalar types
            elif isinstance(field1, (int, float, long, complex, np.number)):
                if np.float32(field1) - np.float32(field2) > 0:
                    print_not_equal_message(
                        attributes_dict[attribute], field1, field2, logger
                    )
                    equal = False
            else:
                equal = assert_equal_objects(field1, field2, logger=logger)
        except BaseException:
            try:
                logger.warning(
                    "Comparing str(objects) for field "
                    + attributes_dict[attribute]
                    + " because there was an error!",
                    logger,
                )
                if np.any(str(field1) != str(field2)):
                    print_not_equal_message(
                        attributes_dict[attribute], field1, field2, logger
                    )
                    equal = False
            except BaseException:
                raise_value_error(
                    "ValueError: Something went wrong when trying to compare "
                    + attributes_dict[attribute]
                    + " !",
                    logger,
                )

    if equal:
        return True
    else:
        return False


def assert_arrays(params, shape=None, transpose=False):
    # type: (object, object) -> object
    if shape is None or not (
        isinstance(shape, tuple)
        and len(shape) in range(3)
        and np.all([isinstance(s, (int, np.int)) for s in shape])
    ):
        shape = None
        shapes = []  # list of all unique shapes
        n_shapes = []  # list of all unique shapes' frequencies
        size = 0  # initial shape
    else:
        size = shape_to_size(shape)

    for ip in range(len(params)):
        # Convert all accepted types to np arrays:
        if isinstance(params[ip], np.ndarray):
            pass
        elif isinstance(params[ip], (list, tuple)):
            # assuming a list or tuple of symbols...
            params[ip] = np.array(params[ip]).astype(type(params[ip][0]))
        elif isinstance(params[ip], (float, int, long, complex, np.number)):
            params[ip] = np.array(params[ip])
        else:
            try:
                import sympy
            except BaseException:
                raise_import_error("sympy import failed")
            if isinstance(params[ip], tuple(sympy.core.all_classes)):
                params[ip] = np.array(params[ip])
            else:
                raise_value_error(
                    "Input "
                    + str(params[ip])
                    + " of type "
                    + str(type(params[ip]))
                    + " is not numeric, "
                    "of type np.ndarray, nor Symbol"
                )
        if shape is None:
            # Only one size > 1 is acceptable
            if params[ip].size != size:
                if size > 1 and params[ip].size > 1:
                    raise_value_error("Inputs are of at least two distinct sizes > 1")
                elif params[ip].size > size:
                    size = params[ip].size
            # Construct a kind of histogram of all different shapes of the
            # inputs:
            ind = np.array([(x == params[ip].shape) for x in shapes])
            if np.any(ind):
                ind = np.where(ind)[0]
                # TODO: handle this properly
                n_shapes[int(ind)] += 1
            else:
                shapes.append(params[ip].shape)
                n_shapes.append(1)
        else:
            if params[ip].size > size:
                raise_value_error(
                    "At least one input is of a greater size than the one given!"
                )

    if shape is None:
        # Keep only shapes of the correct size
        ind = np.array([shape_to_size(s) == size for s in shapes])
        shapes = np.array(shapes)[ind]
        n_shapes = np.array(n_shapes)[ind]
        # Find the most frequent shape
        ind = np.argmax(n_shapes)
        shape = tuple(shapes[ind])

    if transpose and len(shape) > 1:
        if (transpose is "horizontal" or "row" and shape[0] > shape[1]) or (
            transpose is "vertical" or "column" and shape[0] < shape[1]
        ):
            shape = list(shape)
            temp = shape[1]
            shape[1] = shape[0]
            shape[0] = temp
            shape = tuple(shape)

    # Now reshape or tile when necessary
    for ip in range(len(params)):
        try:
            if params[ip].shape != shape:
                if params[ip].size in [0, 1]:
                    params[ip] = np.tile(params[ip], shape)
                else:
                    params[ip] = np.reshape(params[ip], shape)
        except BaseException:
            # TODO: maybe make this an explicit message
            logger.info("\n\nwhat the fuck??")

    if len(params) == 1:
        return params[0]
    else:
        return tuple(params)


def copy_object_attributes(
    obj1, obj2, attr1, attr2=None, deep_copy=False, check_none=False
):
    attr1 = ensure_list(attr1)
    if attr2 is None:
        attr2 = attr1
    else:
        attr2 = ensure_list(attr2)
    if deep_copy:

        def fcopy(a1, a2):
            return setattr(obj2, a2, deepcopy(getattr(obj1, a1)))

    else:

        def fcopy(a1, a2):
            return setattr(obj2, a2, getattr(obj1, a1))

    if check_none:
        for a1, a2 in zip(attr1, attr2):
            if getattr(obj2, a2) is None:
                fcopy(a1, a2)
    else:
        for a1, a2 in zip(attr1, attr2):
            fcopy(a1, a2)
    return obj2
