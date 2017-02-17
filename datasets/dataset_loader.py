# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 13:09:27 2017

@author: sakurai
"""

import random

import numpy as np
from chainer.dataset import DatasetMixin
from fuel.datasets import H5PYDataset

import cars196_dataset
import cub200_2011_dataset
import online_products_dataset
from my_iterators import SerialIterator, MultiprocessIterator
from indexes_samplers import NPairMCIndexesSampler


class RandomCropFlipDataset(DatasetMixin):
    '''
    args:
       images (4D ndarray):
           bchw-shaped uint8 array
       labels (1D ndarray):
           array of class labels
       crop_size (tuple):
           height and width as tuple of integers
    '''
    def __init__(self, dataset, crop_size=(224, 224)):
        if isinstance(dataset, H5PYDataset):
            length = dataset.num_examples
            is_fuel = True
        elif isinstance(dataset, tuple):
            length = len(dataset[0])
            if len(dataset[1]) != length:
                raise ValueError("images and labels must have the same length")
            is_fuel = False
        else:
            ValueError("datasets must be H5PYDataset or a tuple of two "
                       "ndarrays, i.e. (images, labels)")
        self._dataset = dataset
        self._length = length
        self._crop_size = crop_size
        self._is_fuel = is_fuel

    def __len__(self):
        return self._length

    def get_example(self, i):
        if self._is_fuel:
            image, label = self._dataset[i]
        else:
            image, label = [array[i] for array in self._dataset]

        # image of chw-shape
        _, h, w = image.shape
        crop_size_h, crop_size_w = self._crop_size
        # Randomly crop a region and flip the image
        top = random.randint(0, h - crop_size_h - 1)
        left = random.randint(0, w - crop_size_w - 1)
        if random.randint(0, 1):
            image = image[:, :, ::-1]
        bottom = top + crop_size_h
        right = left + crop_size_w

        image = image[:, top:bottom, left:right]
        image = image.astype(np.float32) / 255.0  # Scale to [0, 1]

        return image, label

    def get_labels(self):
        dataset = self._dataset
        if self._is_fuel:
            for i, a_sample in enumerate(dataset[0]):
                if a_sample.ndim == 0 or 1 in a_sample.shape:
                    label_axis = i
                    break
            if dataset.load_in_memory:
                return dataset.data_sources[label_axis]
            else:
                num_examples = len(self)
                batch_size = 100
                labels = []
                for i in range(0, num_examples, batch_size):
                    i_end = min(i + batch_size, num_examples)
                    labels.append(dataset[i:i_end][label_axis])
                return np.concatenate(labels)
        else:
            return dataset[1]


def make_n_pairs_mc_iterator(dataset, batch_size, multiprocess=False,
                             repeat=True):
    chainer_dataset = RandomCropFlipDataset(dataset)
    num_batches = len(chainer_dataset) / batch_size
    labels = chainer_dataset.get_labels()
    order_sampler = NPairMCIndexesSampler(labels, batch_size, num_batches)
    if multiprocess:
        it = MultiprocessIterator(chainer_dataset, batch_size, repeat=repeat,
                                  n_processes=3, order_sampler=order_sampler)
    else:
        it = SerialIterator(chainer_dataset, batch_size, repeat=repeat,
                            order_sampler=order_sampler)
    return it


def make_simple_iterator(dataset, batch_size, repeat=False, shuffle=False,
                         multiprocess=False):
    chainer_dataset = RandomCropFlipDataset(dataset)
    if multiprocess:
        it = MultiprocessIterator(chainer_dataset, batch_size, repeat=repeat,
                                  shuffle=shuffle, n_processes=3)
    else:
        it = SerialIterator(chainer_dataset, batch_size, repeat=repeat,
                            shuffle=shuffle)
    return it


def make_epoch_iterator(dataset, batch_size, multiprocess=False):
    return make_simple_iterator(dataset, batch_size, multiprocess)


def get_iterators(batch_size=50, dataset='cars196', method='n_pairs_mc',
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
        dataset_class = cars196_dataset.Cars196Dataset
    elif dataset == 'cub200_2011':
        dataset_class = cub200_2011_dataset.Cub200_2011Dataset
    elif dataset == 'products':
        dataset_class = online_products_dataset.OnlineProductsDataset
    else:
        raise ValueError(
            "`dataset` must be 'cars196', 'cub200_2011' or 'products'.")

    train = dataset_class(['train'], load_in_memory=load_in_memory)
    test = dataset_class(['test'], load_in_memory=load_in_memory)

    if method == 'n_pairs_mc':
        iter_train = make_n_pairs_mc_iterator(train, batch_size)
    elif method == 'clustering':
        iter_train = make_simple_iterator(train, batch_size,
                                          repeat=True, shuffle=True)
    else:
        raise ValueError("`method` must be 'n_pairs_mc' or 'clustering'.")
    iter_train_eval = make_epoch_iterator(train, batch_size)
    iter_test = make_epoch_iterator(test, batch_size)
    return iter_train, iter_train_eval, iter_test


if __name__ == '__main__':
    batch_size = 50
    dataset = 'cub200_2011'  # 'cars196' or 'cub200_2011' or 'products'
    method = 'n_pairs_mc'  # 'n_pairs_mc' or 'clustering'
    load_in_memory = False

    it = get_iterators(batch_size, dataset, method, load_in_memory)
    it_train, it_train_eval, it_test = it
    batch = next(it_train)
