# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 00:15:50 2017

@author: sakurai
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder

from datasets import get_cars196_streams


class DataProvider(object):
    def __init__(self, stream, batch_size):
        self._stream = stream
        if hasattr(stream, 'data_stream'):
            data_stream = stream.data_stream
        else:
            data_stream = stream
        self.num_examples = data_stream.dataset.num_examples

        labels = [stream.get_data([i])[1][0] for i in range(self.num_examples)]
        labels = np.ravel(labels)

        self._label_encoder = LabelEncoder().fit(labels)
        self._labels = np.array(labels)
        self._classes = self._label_encoder.classes_
        self.num_classes = len(self._classes)
        assert batch_size <= self.num_classes, (
               "batch_size must not be greather than the number of classes"
               " (i.e. batch_size <= {})".format(self.num_classes))
        self.batch_size = batch_size

        self._class_to_indexes = []
        for c in self._classes:
            self._class_to_indexes.append(
                np.argwhere(self._labels == c).ravel())

    def next(self):
        anchor_indexes, positive_indexes = self._generate_indexes()
        indexes = anchor_indexes + positive_indexes
        return self._stream.get_data(indexes)

    def _generate_indexes(self):
        random_classes = np.random.choice(
            self.num_classes, self.batch_size, False)
        anchor_indexes = []
        positive_indexes = []
        for c in random_classes:
            a, p = np.random.choice(self._class_to_indexes[c], 2, False)
            anchor_indexes.append(a)
            positive_indexes.append(p)
        return anchor_indexes, positive_indexes


if __name__ == '__main__':
    train, _ = get_cars196_streams(load_in_memory=True)
    provider = DataProvider(train, 25)

    anchor_indexes, positive_indexes = provider._generate_indexes()
    print anchor_indexes
    print positive_indexes
    a_classes = provider._labels[anchor_indexes]
    p_classes = provider._labels[positive_indexes]
    print np.all(a_classes == p_classes)

    x, c = provider.next()
