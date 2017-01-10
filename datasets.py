# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 14:01:35 2017

@author: sakurai
"""

import numpy as np
from fuel.streams import DataStream
from fuel.transformers.image import RandomFixedSizeCrop

from cars196_dataset import Cars196Dataset
from iteration_schemes import NPairLossScheme


def make_random_cropped_stream(dataset, batch_size, crop_size):
    tmp_stream = DataStream(dataset)
    labels = [tmp_stream.get_data([i])[1] for i in range(dataset.num_examples)]
    labels = np.ravel(labels)
    iteration_scheme = NPairLossScheme(labels, batch_size)

    stream = RandomFixedSizeCrop(
        DataStream(dataset, iteration_scheme=iteration_scheme),
        (crop_size, crop_size), which_sources=("images"))
    return stream


def get_cars196_streams(batch_size, crop_size=227, load_in_memory=False):
    train_dataset = Cars196Dataset(['train'], load_in_memory=load_in_memory)
    train_stream = make_random_cropped_stream(
        train_dataset, batch_size, crop_size)

    test_dataset = Cars196Dataset(['test'], load_in_memory=load_in_memory)
    test_stream = make_random_cropped_stream(
        test_dataset, batch_size, crop_size)

    return train_stream, test_stream


if __name__ == '__main__':
    batch_size = 50
    train, test = get_cars196_streams(batch_size, load_in_memory=False)
#    train.get_data([0, 1, 2])
    it = train.get_epoch_iterator()
    x, c = next(it)
    print c.ravel()[:batch_size / 2].tolist()
    print c.ravel()[batch_size / 2:].tolist()
