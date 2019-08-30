import os

import h5py
import numpy as np
import json

from eztrack.eegio.writers import MetadataSchema

"""
0. create writer for pyarrow parquet
- write resulting computations
	-> store array
	-> store metadata
	

Reference: 
1. https://arrow.apache.org/docs/python/parquet.html
"""

# the data type for the numpy array datasets
DTYPE = np.float64


class BaseContainer(object):
    def __init__(self, fragmat, pertmat, ltvmat, **kwargs):
        self.metadata = kwargs
        # datasets = ensure_list(datasets)

        # for data in datasets:
        self.fragmat = fragmat
        self.pertmat = pertmat
        self.ltvmat = ltvmat

        metadataschema = MetadataSchema()
        metadataschema.schemamapcheck(self.metadata)

    def __str__(self):
        return "Dataset container with: fragmat, pertmat, ltvmat &" \
               " accompanying metadata."

    @property
    def shape(self):
        return self.fragmat.shape

    @property
    def datasets(self):
        return {
            'fragmat': self.fragmat,
            'pertmat': self.pertmat,
            'ltvmat': self.ltvmat
        }

    def set_metadata(self, metadata):
        self.metadata = metadata

        metadataschema = MetadataSchema()
        metadataschema.schemamapcheck(metadata)

    def load_into_mem(self, name='fragmat'):
        if name == 'fragmat':
            data = self.fragmat[...]
        elif name == 'pertmat':
            data = self.pertmat[...]
        elif name == 'ltvmat':
            data = self.ltvmat[...]
        return data


class BaseWriter(object):
    def __init__(self, filepath, container=None, dtype=DTYPE):
        # if not type(container) == BaseContainer:
        #     raise AttributeError("All containers passed to base writer, should be of"
        #                          "the form of BaseContainer in eegio.writers.base! You passed in "
        #                          "{}".format(type(container)))
        self.filepath = filepath
        self.container = container
        self.method = 'hdf'
        self.dtype = dtype

    @property
    def sizebytes(self):
        return os.path.getsize(self.filepath)

    def _creategroup(self, name, filepath):
        with h5py.File(filepath, mode='a') as outfile:
            if not name in outfile.keys():
                group = outfile.create_group(name=name,
                                             track_order=True)
            else:
                group = outfile[name]
            return group

    def load_file(self):
        hfile = h5py.File(self.filepath, mode='r')
        return hfile

    def load_patient(self, patient_id):
        pass

    def load_dataset_types(self, datatype):
        pass

    def load_metadata(self, patient_id):
        pass

    def save_patientgroup(self, dataset_id, datasetnames, datasetlist, metadata):
        """
        Main method to save patients in a group!

        To load string dataset metadata use:

            --> ast.literal_eval(metadatastr)

        TODO:
        - set separate metadata for each dataset.

        :param name:
        :param dataset_id:
        :param datasetnames:
        :param datasetlist:
        :param metadata:
        :return:
        """
        print("Saving ", dataset_id)

        # patiengroup = self._creategroup(name, self.filepath)
        with h5py.File(self.filepath, mode='a') as patientgroup:
            # get patient group
            # if not name in outfile.keys():
            #     patientgroup = outfile.create_group(name=name,
            #                                         track_order=False)
            # else:
            #     patientgroup = outfile[name]

            # get datasetid group
            if not dataset_id in patientgroup.keys():
                datasetid_group = patientgroup.create_group(dataset_id,
                                                            track_order=False)
            else:
                datasetid_group = patientgroup[dataset_id]

            # add datasets under names
            for idx, dataset in enumerate(datasetlist):
                datasetname = datasetnames[idx]
                # metadata = metadatalist[idx]

                if datasetname not in datasetid_group.keys():
                    dataset = datasetid_group.create_dataset(name=datasetname,
                                                             shape=dataset.shape,
                                                             data=dataset,
                                                             dtype=DTYPE,
                                                             compression='gzip')
            # save one metadata per dataset id
            metadataschema = MetadataSchema()
            metadataschema.schemamapcheck(metadata)
            metadata = metadataschema.trimkeys(metadata)
            metadatasetname = 'metadata'
            dataset = datasetid_group.create_dataset(name=metadatasetname,
                                                     data=json.dumps(metadata))
            # self.set_metadata_attrs(datasetid_group, metadata)

            print("Finished writing to the datasets")

    def save_datasetgroup(self, dataset_id, datasetnames, datasetlist, patientgroup=None):
        if patientgroup is not None:
            group = patientgroup
        else:
            group = self._creategroup(dataset_id, self.filepath)

        for idx, dataset in enumerate(datasetlist):
            datasetname = datasetnames[idx]
            dataset = group.create_dataset(name=datasetname,
                                           shape=dataset.shape,
                                           data=dataset,
                                           dtype=DTYPE,
                                           compression='gzip')

    def load_metadata_attrs(self, dataset, metadata={}):
        """
        Helper method to load in metadata attributes and helps convert
        lists of strings as bytes into unicode.

        :param dataset:
        :param metadata:
        :return:
        """
        for key in dataset.attrs.keys():
            try:
                if isinstance(dataset.attrs[key], np.ndarray):
                    metadata[key] = np.array(
                        [item.decode('utf-8') for item in dataset.attrs[key]])
                    # print(dataset.attrs[key])
                else:
                    metadata[key] = dataset.attrs[key]
            except:
                metadata[key] = dataset.attrs[key]
        return metadata

    def set_metadata_attrs(self, dataset, metadata):
        """
        Helper method to set metadata during the saving of attributes.

        :param dataset:
        :param metadata:
        :return:
        """
        # if not type(container) == BaseContainer:
        #     raise AttributeError("All containers passed to base writer, should be of"
        #                          "the form of BaseContainer in eegio.writers.base! You passed in "
        #                          "{}".format(type(container)))

        for key in metadata.keys():
            # print(key, type(container.metadata[key]))
            if isinstance(metadata[key], list) and metadata[key]:
                if isinstance(metadata[key][0], str):
                    metadata[key] = np.string_(metadata[key])
            dataset.attrs[key] = metadata[key]

    def read_container(self, filepath, name='fragility_results'):
        datasetnames = [
            'fragmat',
            'pertmat',
            'ltvmat'
        ]
        with h5py.File(filepath, mode='r') as infile:
            print("Reading file {}".format(infile))

            group = infile[name]

            # access the h5py groups
            fragmat = group['fragmat'][...]
            pertmat = group['pertmat'][...]
            ltvmat = group['ltvmat'][...]

            # load in the metadata
            metadata = self.load_metadata_attrs(group)

        container = BaseContainer(fragmat, pertmat, ltvmat, **metadata)
        return container

    def save_container(self, group_name='fragility_results'):
        if self.method == 'hd5':
            with h5py.File(self.filepath, mode='w') as outfile:
                print("Saving {} as {}".format(
                    self.filepath,
                    outfile
                ))
                group = outfile.create_group(name=group_name,
                                             track_order=True)
                for idx, (name, data) in enumerate(self.container.datasets.items()):
                    dataset = group.create_dataset(name=name,
                                                   shape=data.shape,
                                                   data=data,
                                                   dtype=DTYPE,
                                                   compression='gzip')
                self.set_metadata_attrs(group, self.container)
