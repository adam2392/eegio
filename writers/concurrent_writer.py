from eztrack.eegio.writers import BaseWriter
import zarr
# import pyarrow as pa


class ConcurrentWriter(BaseWriter):
    """
    Class for concurrent writing to support hdf files being written to from multiple processes.

    This is mainly to allow scaling of algorithm writing onto a HPC cluster.
    """

    def __init__(self, filepath, datashape, chunkshape):
        super(ConcurrentWriter, self).__init__(filepath)

        # initialize datashape
        self.datashape = datashape
        self.chunkshape = chunkshape

        # inialize synchronizer
        filepathsync = filepath + '.sync'
        self.synchronizer = zarr.ProcessSynchronizer(filepathsync)

    def _init_file(self):
        # try creating the group first
        root = zarr.group()

        self.z = zarr.open_array(self.filepath, mode='w',
                                 shape=self.datashape,
                                 chunks=self.chunkshape,
                                 dtype=self.dtype,
                                 synchronizer=self.synchronizer)

    def write_vector(self, vec, index):
        self.z[index, :] = vec

    def write_arr(self, arr):
        self.z = arr


class ConcurrentArrowWriter(BaseWriter):
    def __init__(self, filepath, datashape, chunkshape):
        super(ConcurrentArrowWriter, self).__init__(filepath)

        # initialize datashape
        self.datashape = datashape
        self.chunkshape = chunkshape

    def define_parquet_schema(self):

    def write_arr(self, arr):
        arr = pa.array(arr)
