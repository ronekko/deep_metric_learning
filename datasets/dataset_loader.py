# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 13:09:27 2017

@author: sakurai
"""

import random

import numpy as np
from chainer.dataset import DatasetMixin

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
    def __init__(self, images, labels, crop_size=(224, 224)):
        length = len(images)
        if len(labels) != length:
            raise ValueError("images and labels must have the same length")
        self._images = images
        self._labels = labels
        self._length = length
        self._crop_size = crop_size

    def __len__(self):
        return self._length

    def get_example(self, i):
        image = self._images[i]  # a chw-shaped image
        label = self._labels[i]

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


def make_n_pairs_mc_iterator(x, labels, batch_size, multiprocess=False,
                             repeat=True):
    dataset = RandomCropFlipDataset(x, labels)
    num_batches = len(dataset) / batch_size
    order_sampler = NPairMCIndexesSampler(labels, batch_size, num_batches)
    if multiprocess:
        it = MultiprocessIterator(dataset, batch_size, repeat=repeat,
                                  n_processes=3, order_sampler=order_sampler)
    else:
        it = SerialIterator(dataset, batch_size, repeat=repeat,
                            order_sampler=order_sampler)
    return it


def make_simple_iterator(x, labels, batch_size, repeat=False, shuffle=False,
                         multiprocess=False):
    dataset = RandomCropFlipDataset(x, labels)
    if multiprocess:
        it = MultiprocessIterator(dataset, batch_size, repeat=repeat,
                                  shuffle=shuffle, n_processes=3)
    else:
        it = SerialIterator(dataset, batch_size, repeat=repeat,
                            shuffle=shuffle)
    return it


def make_epoch_iterator(x, labels, batch_size, multiprocess=False):
    return make_simple_iterator(x, labels, batch_size, multiprocess)


def get_iterators(batch_size=50, dataset='cars196', method='n_pairs_mc'):
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
        dataset_module = cars196_dataset
    elif dataset == 'cub200_2011':
        dataset_module = cub200_2011_dataset
    elif dataset == 'products':
        dataset_module = online_products_dataset
    else:
        raise ValueError(
            "`dataset` must be 'cars196', 'cub200_2011' or 'products'.")

    train, test = dataset_module.load_as_ndarray(['train', 'test'])
    if method == 'n_pairs_mc':
        iter_train = make_n_pairs_mc_iterator(train[0], train[1], batch_size)
    elif method == 'clustering':
        iter_train = make_simple_iterator(train[0], train[1], batch_size,
                                          repeat=True, shuffle=True)
    else:
        raise ValueError("`method` must be 'n_pairs_mc' or 'clustering'.")
    iter_train_eval = make_epoch_iterator(train[0], train[1], batch_size)
    iter_test = make_epoch_iterator(test[0], test[1], batch_size)
    return iter_train, iter_train_eval, iter_test


if __name__ == '__main__':
    multiprocess = False
    batch_size = 50
    train = cars196_dataset.load_as_ndarray(['train'])[0]
    x, labels = train
    dataset = RandomCropFlipDataset(x, labels)
    num_batches = len(dataset) / batch_size
    order_sampler = NPairMCIndexesSampler(labels, batch_size, num_batches)
    if multiprocess:
        it = MultiprocessIterator(dataset, batch_size, repeat=True,
                                  n_processes=3,
                                  order_sampler=order_sampler)
    else:
        it = SerialIterator(dataset, batch_size, repeat=True,
                            order_sampler=order_sampler)

    for i in range(1):
        batch = next(it)
        l = np.ravel([pair[1] for pair in batch]).tolist()
        print i
        print l[:batch_size/2]
        print l[batch_size/2:]
        print
