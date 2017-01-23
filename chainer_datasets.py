# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 13:09:27 2017

@author: sakurai
"""

import numpy as np
import matplotlib.pyplot as plt

from chainer.datasets import TupleDataset

import cars196_dataset
from my_iterators import SerialIterator
from indexes_samplers import NPairMCIndexesSampler


if __name__ == '__main__':
    batch_size = 50
    train = cars196_dataset.load_as_ndarray(['train'])[0]
    x, labels = train
    dataset = TupleDataset(x, labels)
    num_batches = len(dataset) / batch_size
    order_sampler = NPairMCIndexesSampler(labels, batch_size, num_batches)
    it = SerialIterator(dataset, batch_size, True, order_sampler=order_sampler)

    for i in range(num_batches):
        batch = next(it)
        l = np.ravel([pair[1] for pair in batch]).tolist()
        print i
        print l[:batch_size/2]
        print l[batch_size/2:]
        print
