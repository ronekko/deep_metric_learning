# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 19:02:49 2016

@author: sakurai
"""

from fuel.datasets import H5PYDataset
from fuel.utils import find_in_data_path
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream


class Cars196Dataset(H5PYDataset):

    _filename = 'cars196/cars196.hdf5'

    def __init__(self, which_sets, **kwargs):
        super(Cars196Dataset, self).__init__(
            file_or_path=find_in_data_path(self._filename),
            which_sets=which_sets, **kwargs)

if __name__ == '__main__':
    dataset = Cars196Dataset(['train'])

    st = DataStream(
        dataset, iteration_scheme=SequentialScheme(dataset.num_examples, 1))
    it = st.get_epoch_iterator()
    it.next()
