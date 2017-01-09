# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 14:01:35 2017

@author: sakurai
"""


from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from random_fixed_size_crop_mod import RandomFixedSizeCrop

from cars196_dataset import Cars196Dataset


def make_random_cropped_stream(dataset, crop_size):
    # This scheme is dummy, since DataStream requires an iteration_scheme for
    # DataStream.produces_examples to be False in the constructor.
    dummy_scheme = SequentialScheme(1, 1)

    stream = RandomFixedSizeCrop(
        DataStream(dataset, iteration_scheme=dummy_scheme),
        (crop_size, crop_size), which_sources=("images"))
    return stream


def get_cars196_streams(crop_size=227, load_in_memory=False):
    train_dataset = Cars196Dataset(['train'], load_in_memory=load_in_memory)
    train_stream = make_random_cropped_stream(train_dataset, crop_size)

    test_dataset = Cars196Dataset(['test'], load_in_memory=load_in_memory)
    test_stream = make_random_cropped_stream(test_dataset, crop_size)

    return train_stream, test_stream


if __name__ == '__main__':
    train, test = get_cars196_streams(load_in_memory=True)
    train.get_data([0, 1, 2])
