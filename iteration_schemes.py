# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:51:07 2017

@author: sakurai
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from fuel.schemes import BatchSizeScheme, ShuffledScheme


class NPairLossScheme(BatchSizeScheme):
    def __init__(self, labels, batch_size):
        self._labels = np.array(labels)
        self._label_encoder = LabelEncoder().fit(self._labels)
        self._classes = self._label_encoder.classes_
        self.num_classes = len(self._classes)
        assert batch_size % 2 == 0, ("batch_size must be even number.")
        assert batch_size <= self.num_classes * 2, (
               "batch_size must not exceed twice the number of classes"
               "(i.e. set batch_size <= {}).".format(self.num_classes * 2))
        self.batch_size = batch_size

        self._class_to_indexes = []
        for c in self._classes:
            self._class_to_indexes.append(
                np.argwhere(self._labels == c).ravel())

    def __iter__(self):
        return self

    def next(self):
        anchor_indexes, positive_indexes = self._generate_indexes()
        indexes = anchor_indexes + positive_indexes
        return indexes

    def _generate_indexes(self):
        random_classes = np.random.choice(
            self.num_classes, self.batch_size / 2, False)
        anchor_indexes = []
        positive_indexes = []
        for c in random_classes:
            a, p = np.random.choice(self._class_to_indexes[c], 2, False)
            anchor_indexes.append(a)
            positive_indexes.append(p)
        return anchor_indexes, positive_indexes

    def get_request_iterator(self):
        return self


if __name__ == '__main__':
    batch_size = 20
#    s = ShuffledScheme(10, 3)
    labels = sum([[i]*10 for i in range(10)], [])
    s = NPairLossScheme(labels, batch_size)
    i = 0
    for indexes in s.get_request_iterator():
        a_indexes = indexes[:batch_size / 2]
        p_indexes = indexes[batch_size / 2:]
        print a_indexes
        print p_indexes
        print
        i += 1
        if i > 5:
            break
