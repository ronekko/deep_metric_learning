# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 21:16:43 2017

@author: ryuhei
"""

import numpy as np

from chainer.datasets import TupleDataset
from sklearn.preprocessing import LabelEncoder
from my_iterators import SerialIterator, MultiprocessIterator


class NPairMCIndexesSampler(object):
    def __init__(self, labels, batch_size, num_batches):
        assert len(labels) >= (batch_size * num_batches), (
            "batch_size * num_batches must not exceed the number of examples")
        self._labels = np.array(labels).flatten()
        self._label_encoder = LabelEncoder().fit(self._labels)
        self._classes = self._label_encoder.classes_
        self.num_classes = len(self._classes)
        assert batch_size % 2 == 0, ("batch_size must be even number.")
        assert batch_size <= self.num_classes * 2, (
            "batch_size must not exceed twice the number of classes"
            "(i.e. set batch_size <= {}).".format(self.num_classes * 2))
        self.batch_size = batch_size
        self.num_batches = num_batches

        self._class_to_indexes = []
        for c in self._classes:
            self._class_to_indexes.append(
                np.argwhere(self._labels == c).ravel())

    def __call__(self):
        indexes = []
        for _ in range(self.num_batches):
            random_classes = np.random.choice(
                self.num_classes, self.batch_size / 2, False)
            anchor_indexes = []
            positive_indexes = []
            for c in random_classes:
                a, p = np.random.choice(self._class_to_indexes[c], 2, False)
                anchor_indexes.append(a)
                positive_indexes.append(p)
            indexes.append(anchor_indexes)
            indexes.append(positive_indexes)
        return np.concatenate(indexes)


if __name__ == '__main__':
    batch_size = 10
    num_batches = 5
    repeat = True
    multiprocess = True

    labels = np.array(sum([[i]*10 for i in range(10)], []))
    num_examples = len(labels)
    x = np.arange(num_examples)
    dataset = TupleDataset(x, labels)

    indexes_sampler = NPairMCIndexesSampler(labels, batch_size, num_batches)
    if multiprocess:
        it = MultiprocessIterator(dataset, batch_size, repeat=repeat,
                                  shuffle=True)
#                                  order_sampler=indexes_sampler)
    else:
        it = SerialIterator(dataset, batch_size, repeat=repeat,
                            shuffle=True)
#                            order_sampler=indexes_sampler)

    for i in range(num_batches*2):
        batch = next(it)
        print len(batch)
        print batch[:batch_size/2]
        print batch[batch_size/2:]
        print

    if multiprocess:
        it.finalize()
