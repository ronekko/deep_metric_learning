# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 14:01:35 2017

@author: sakurai
"""


from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from random_fixed_size_crop_mod import RandomFixedSizeCrop

from cars196_dataset import Cars196Dataset


def get_cars196_streams(crop_size=227, load_in_memory=False):
    # This scheme is dummy, since DataStream requires an iteration_scheme for
    # DataStream.produces_examples to be False in the constructor.
    dummy_scheme = SequentialScheme(1, 1)

    train_stream = RandomFixedSizeCrop(
        DataStream(Cars196Dataset(['train'], load_in_memory=load_in_memory),
                   iteration_scheme=dummy_scheme),
        (crop_size, crop_size), which_sources=("images"))
    test_stream = RandomFixedSizeCrop(
        DataStream(Cars196Dataset(['test'], load_in_memory=load_in_memory),
                   iteration_scheme=dummy_scheme),
        (crop_size, crop_size), which_sources=("images"))
    return train_stream, test_stream


if __name__ == '__main__':
    train, test = get_cars196_streams(load_in_memory=True)
    train.get_data([0, 1, 2])
