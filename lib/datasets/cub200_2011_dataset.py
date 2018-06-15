# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 00:57:05 2017

@author: sakurai
"""

from fuel.datasets import H5PYDataset
from fuel.utils import find_in_data_path
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream


class Cub200_2011Dataset(H5PYDataset):

    _filename = 'cub200_2011/cub200_2011.hdf5'

    def __init__(self, which_sets, **kwargs):
        try:
            path = find_in_data_path(self._filename)
        except IOError as e:
            msg = str(e) + (""".
         You need to download the dataset and convert it to hdf5 before.""")
            raise IOError(msg)
        super(Cub200_2011Dataset, self).__init__(
            file_or_path=path, which_sets=which_sets, **kwargs)


def load_as_ndarray(which_sets=['train', 'test']):
    datasets = []
    for split in which_sets:
        data = Cub200_2011Dataset([split], load_in_memory=True).data_sources
        datasets.append(data)
    return datasets


if __name__ == '__main__':
    dataset = Cub200_2011Dataset(['train'])

    st = DataStream(
        dataset, iteration_scheme=SequentialScheme(dataset.num_examples, 1))
    it = st.get_epoch_iterator()
    next(it)
