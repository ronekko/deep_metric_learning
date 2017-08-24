# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 17:30:49 2017

@author: sakurai
"""

from fuel.datasets import H5PYDataset
from fuel.utils import find_in_data_path
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream


class OnlineProductsDataset(H5PYDataset):

    _filename = 'online_products/online_products.hdf5'

    def __init__(self, which_sets, **kwargs):
        try:
            path = find_in_data_path(self._filename)
        except IOError as e:
            msg = str(e) + (""".
         You need to download the dataset and convert it to hdf5 before.""")
            raise IOError(msg)
        super(OnlineProductsDataset, self).__init__(
            file_or_path=path, which_sets=which_sets, **kwargs)


def load_as_ndarray(which_sets=['train', 'test']):
    datasets = []
    for split in which_sets:
        data = OnlineProductsDataset([split], load_in_memory=True).data_sources
        datasets.append(data)
    return datasets


if __name__ == '__main__':
    dataset = OnlineProductsDataset(['train'])

    st = DataStream(
        dataset, iteration_scheme=SequentialScheme(dataset.num_examples, 1))
    it = st.get_epoch_iterator()
    it.next()
