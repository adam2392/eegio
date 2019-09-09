try:
    H5PY_SUPPORT = True
    import h5py as hdf5
except ImportError:
    H5PY_SUPPORT = False
import os
import zipfile
import warnings
import numpy

from dev.tvb import Connectivity

'''

Load regions functions for eegio analysis module to get structural
connectivity data and load it so that you can get access to

- region labels in cortex
- region xyz coords
- weights between regions (derived from dMRI)
- and much more...

'''


class LoadConn(object):

    def readconnectivity(self, source_file):

        source_full_path = self.try_get_absolute_path(
            "tvb_data.connectivity", source_file)
        result = dict()

        if source_file.endswith(".h5"):

            reader = H5Reader(source_full_path)

            # result['weights'] = reader.read_field("weights")
            result['centres'] = reader.read_field("centres")
            result['region_labels'] = reader.read_field("region_labels")
            result['orientations'] = reader.read_optional_field("orientations")
            result['cortical'] = reader.read_optional_field("cortical")
            result['hemispheres'] = reader.read_field("hemispheres")
            result['areas'] = reader.read_optional_field("areas")
            # result['tract_lengths'] = reader.read_field("tract_lengths")

        else:
            reader = ZipReader(source_full_path)

            # result['weights'] = reader.read_array_from_file("weights")
            result['centres'] = reader.read_array_from_file(
                "centres", use_cols=(1, 2, 3))
            result['region_labels'] = reader.read_array_from_file(
                "centres", dtype=numpy.str, use_cols=(0,))
            result['orientations'] = reader.read_optional_array_from_file(
                "average_orientations")
            result['cortical'] = reader.read_optional_array_from_file(
                "cortical", dtype=numpy.bool)
            result['hemispheres'] = reader.read_optional_array_from_file(
                "hemispheres", dtype=numpy.bool)
            result['areas'] = reader.read_optional_array_from_file("areas")
            # result['tract_lengths'] = reader.read_array_from_file(
            #     "tract_lengths")

        conn = self.convert_to_obj(source_file, result)
        return conn

    def convert_to_obj(self, path, result):
        # weights = result['weights']
        region_centres = result['centres']
        region_labels = result['region_labels']
        orientations = result['orientations']
        hemispheres = result['hemispheres']
        areas = result['areas']
        # tract_lengths = result['tract_lengths']

        conn = Connectivity(
            path,
            # weights,
            # tract_lengths,
            region_labels,
            region_centres,
            hemispheres,
            orientations)
        return conn

    def try_get_absolute_path(self, relative_module, file_suffix):
        """
        :param relative_module: python module to be imported. When import of this fails, we will return the file_suffix
        :param file_suffix: In case this is already an absolute path, return it immediately,
            otherwise append it after the module path
        :return: Try to build an absolute path based on a python module and a file-suffix
        """
        result_full_path = file_suffix

        if not os.path.isabs(file_suffix):

            try:
                module_import = __import__(
                    relative_module, globals(), locals(), ["__init__"])
                result_full_path = os.path.join(
                    os.path.dirname(
                        module_import.__file__),
                    file_suffix)

            except ImportError:
                raise Exception(
                    "Could not import tvb_data Python module for default data-set!")
        return result_full_path


class H5Reader(object):
    """
    Read one or many numpy arrays from a H5 file.
    """

    def __init__(self, h5_path):
        if H5PY_SUPPORT:
            self.hfd5_source = hdf5.File(h5_path, 'r', libver='latest')
        else:
            warnings.warning(
                "You need h5py properly installed in order to load from a HDF5 source.")

    def read_field(self, field, ):

        try:
            return self.hfd5_source['/' + field][()]
        except Exception:
            raise RuntimeError("Could not read from %s field" % field)

    def read_optional_field(self, field):
        try:
            return self.read_field(field)
        except Exception:
            return numpy.array([])


class FileReader(object):
    """
    Read one or multiple numpy arrays from a text/bz2 file.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.file_stream = file_path

    def read_array(self, dtype=numpy.float64, skip_rows=0,
                   use_cols=None, matlab_data_name=None):
        try:
            # Try to read H5:
            if self.file_path.endswith('.h5'):
                return numpy.array([])

            # Try to read NumPy:
            if self.file_path.endswith(
                    '.txt') or self.file_path.endswith('.bz2'):
                return self._read_text(
                    self.file_stream, dtype, skip_rows, use_cols)

            if self.file_path.endswith(
                    '.npz') or self.file_path.endswith(".npy"):
                return numpy.load(self.file_stream)

            # Try to read Matlab format:
            return self._read_matlab(self.file_stream, matlab_data_name)

        except Exception:
            raise Exception(
                "Could not read from %s file" %
                self.file_path)

    def _read_text(self, file_stream, dtype, skip_rows, use_cols):

        array_result = numpy.loadtxt(
            file_stream,
            dtype=dtype,
            skiprows=skip_rows,
            usecols=use_cols)
        return array_result

    def _read_matlab(self, file_stream, matlab_data_name=None):

        if self.file_path.endswith(".mtx"):
            return scipy_io.mmread(file_stream)

        if self.file_path.endswith(".mat"):
            matlab_data = scipy_io.matlab.loadmat(file_stream)
            return matlab_data[matlab_data_name]

    def read_gain_from_brainstorm(self):

        if not self.file_path.endswith('.mat'):
            raise Exception(
                "Brainstorm format is expected in a Matlab file not %s" %
                self.file_path)

        mat = scipy_io.loadmat(self.file_stream)
        expected_fields = ['Gain', 'GridLoc', 'GridOrient']

        for field in expected_fields:
            if field not in mat.keys():
                raise Exception(
                    "Brainstorm format is expecting field %s" %
                    field)

        gain, loc, ori = (mat[field] for field in expected_fields)
        return (gain.reshape((gain.shape[0], -1, 3)) * ori).sum(axis=-1)


class ZipReader(object):
    """
    Read one or many numpy arrays from a ZIP archive.
    """

    def __init__(self, zip_path):
        self.zip_archive = zipfile.ZipFile(zip_path)

    def read_array_from_file(self, file_name, dtype=numpy.float64,
                             skip_rows=0, use_cols=None, matlab_data_name=None):

        matching_file_name = None
        for actual_name in self.zip_archive.namelist():
            if file_name in actual_name and not actual_name.startswith(
                    "__MACOSX"):
                matching_file_name = actual_name
                break

        if matching_file_name is None:
            raise Exception("File %r not found in ZIP." % file_name)

        zip_entry = self.zip_archive.open(matching_file_name, 'r')

        if matching_file_name.endswith(".bz2"):
            temp_file = copy_zip_entry_into_temp(zip_entry, matching_file_name)
            file_reader = FileReader(temp_file)
            result = file_reader.read_array(
                dtype, skip_rows, use_cols, matlab_data_name)
            os.remove(temp_file)
            return result

        file_reader = FileReader(matching_file_name)
        file_reader.file_stream = zip_entry
        return file_reader.read_array(
            dtype, skip_rows, use_cols, matlab_data_name)

    def read_optional_array_from_file(self, file_name, dtype=numpy.float64, skip_rows=0,
                                      use_cols=None, matlab_data_name=None):
        try:
            return self.read_array_from_file(
                file_name, dtype, skip_rows, use_cols, matlab_data_name)
        except Exception:
            return numpy.array([])
