# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 00:15:50 2017

@author: sakurai
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder
from fuel.streams import DataStream
from fuel.schemes import BatchSizeScheme, SequentialScheme

from cars196_dataset import Cars196Dataset


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
               " (i.e. set batch_size <= {})".format(self.num_classes))
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
from datasets.cub200_2011_dataset import Cub200_2011Dataset
from datasets.online_products_dataset import OnlineProductsDataset
from random_fixed_size_crop_mod import RandomFixedSizeCrop


def get_streams(batch_size=50, dataset='cars196', method='n_pairs_mc',
                load_in_memory=False):
    '''
    args:
       batch_size (int):
           number of examples per batch
       dataset (str):
           specify the dataset from 'cars196', 'cub200_2011', 'products'.
       method (str):
           batch construction method. Select from 'n_pairs_mc', 'clustering'.
    '''

    if dataset == 'cars196':
        dataset_class = Cars196Dataset
    elif dataset == 'cub200_2011':
        dataset_class = Cub200_2011Dataset
    elif dataset == 'products':
        dataset_class = OnlineProductsDataset
    else:
        raise ValueError(
            "`dataset` must be 'cars196', 'cub200_2011' or 'products'.")

    dataset_train = dataset_class(['train'], load_in_memory=load_in_memory)
    dataset_test = dataset_class(['test'], load_in_memory=load_in_memory)

    if method == 'n_pairs_mc':
        labels = dataset_class(
            ['train'], sources=['targets'], load_in_memory=True).data_sources
        scheme = NPairLossScheme(labels, batch_size)
    elif method == 'clustering':
        scheme = EpochwiseShuffledInfiniteScheme(
            dataset_train.num_examples, batch_size)
    else:
        raise ValueError("`method` must be 'n_pairs_mc' or 'clustering'.")
    stream = DataStream(dataset_train, iteration_scheme=scheme)
    stream_train = RandomFixedSizeCrop(stream, which_sources=('images',),
                                       random_lr_flip=True)

    stream_train_eval = RandomFixedSizeCrop(DataStream(
        dataset_train, iteration_scheme=SequentialScheme(
            dataset_train.num_examples, batch_size)),
        which_sources=('images',), random_lr_flip=True)
    stream_test = RandomFixedSizeCrop(DataStream(
        dataset_test, iteration_scheme=SequentialScheme(
            dataset_test.num_examples, batch_size)),
        which_sources=('images',), random_lr_flip=True)

    return stream_train, stream_train_eval, stream_test


class NPairLossScheme(BatchSizeScheme):
    def __init__(self, labels, batch_size):
        self._labels = np.array(labels).flatten()
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


class EpochwiseShuffledInfiniteScheme(BatchSizeScheme):
    def __init__(self, indexes, batch_size):
        if isinstance(indexes, int):
            indexes = range(indexes)
        if batch_size > len(indexes):
            raise ValueError('batch_size must not be larger than the indexes.')
        if len(indexes) != len(np.unique(indexes)):
            raise ValueError('Items in indexes must be unique.')
        self._original_indexes = np.array(indexes, dtype=np.int).flatten()
        self.batch_size = batch_size
        self._shuffled_indexes = np.array([], dtype=np.int)

    def __iter__(self):
        return self

    def next(self):
        batch_size = self.batch_size

        # if remaining indexes are shorter than batch_size then new shuffled
        # indexes are appended to the remains.
        num_remains = len(self._shuffled_indexes)
        if num_remains < batch_size:
            num_overrun = batch_size - num_remains
            new_shuffled_indexes = self._original_indexes.copy()

            # ensure the batch of indexes from the joint part does not contain
            # duplicate index.
            np.random.shuffle(new_shuffled_indexes)
            overrun = new_shuffled_indexes[:num_overrun]
            next_indexes = np.concatenate((self._shuffled_indexes, overrun))
            while len(next_indexes) != len(np.unique(next_indexes)):
                np.random.shuffle(new_shuffled_indexes)
                overrun = new_shuffled_indexes[:num_overrun]
                next_indexes = np.concatenate(
                    (self._shuffled_indexes, overrun))
            self._shuffled_indexes = np.concatenate(
                (self._shuffled_indexes, new_shuffled_indexes))
        next_indexes = self._shuffled_indexes[:batch_size]
        self._shuffled_indexes = self._shuffled_indexes[batch_size:]
        return next_indexes

    def get_request_iterator(self):
        return self


def make_random_cropped_stream(dataset, batch_size, crop_size):
    tmp_stream = DataStream(dataset)
    labels = [tmp_stream.get_data([i])[1] for i in range(dataset.num_examples)]
    labels = np.ravel(labels)
    iteration_scheme = NPairLossScheme(labels, batch_size)

    stream = RandomFixedSizeCrop(
        DataStream(dataset, iteration_scheme=iteration_scheme),
        (crop_size, crop_size), which_sources=("images"))

    # This line is necessary to work with multiprocessing.
    # The hdf5 file is opened when the first data is fetched and it seem that
    # the file-open must be done in the main process otherwise an error occurs.
    stream.get_epoch_iterator().next()
    return stream


def get_cars196_streams(batch_size, crop_size=227, load_in_memory=False):
    train_dataset = Cars196Dataset(['train'], load_in_memory=load_in_memory)
    train_stream = make_random_cropped_stream(
        train_dataset, batch_size, crop_size)

    test_dataset = Cars196Dataset(['test'], load_in_memory=load_in_memory)
    sequential_scheme = SequentialScheme(test_dataset.num_examples, batch_size)
    test_stream = RandomFixedSizeCrop(DataStream(
        test_dataset, iteration_scheme=sequential_scheme),
        (crop_size, crop_size), which_sources=("images"))

    return train_stream, test_stream


if __name__ == '__main__':
    train, _ = get_cars196_streams(50, load_in_memory=True)
    provider = DataProvider(train, 25)

    anchor_indexes, positive_indexes = provider._generate_indexes()
    print anchor_indexes
    print positive_indexes
    a_classes = provider._labels[anchor_indexes]
    p_classes = provider._labels[positive_indexes]
    print np.all(a_classes == p_classes)

    x, c = provider.next()

#    batch_size = 50
#    train, test = get_cars196_streams(batch_size, load_in_memory=False)
#    it = train.get_epoch_iterator()
#    x, c = next(it)
#    print c.ravel()[:batch_size / 2].tolist()
#    print c.ravel()[batch_size / 2:].tolist()
#
#    it = test.get_epoch_iterator()
#    x, c = next(it)
#    print c.ravel()[:batch_size / 2].tolist()
#    print c.ravel()[batch_size / 2:].tolist()
