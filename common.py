# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 01:55:58 2016

@author: sakurai
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt

import chainer
from chainer import Variable


# average accuracy and distance matrix for test data
def evaluate(model, x_data, id_data, batch_size, train=False):
    cupy = chainer.cuda.cupy
    num_examples = len(x_data)
    num_batches = num_examples / batch_size
    # fprop to calculate distance matrix (not for backprop)
    y_batches = []
    for x_batch_data in np.array_split(x_data, num_batches):
        x_batch = Variable(cupy.asarray(x_batch_data), volatile=not train)
        y_batch = model(x_batch)
        y_batches.append(y_batch.data)
        y_batch = None
    y_data = cupy.concatenate(y_batches)

    D = cupy.empty((num_examples, num_examples))
    splits = np.array_split(np.arange(num_examples), num_batches)
    for indices in splits:
        start = indices[0]
        end = start + len(indices)
        y_batch = y_data[start:end]
        D[start:end] = cupy.sum(
            (cupy.expand_dims(y_batch, 1) - cupy.expand_dims(y_data, 0)) ** 2,
            axis=2)
    D = cupy.sqrt(D).get()

    softs = []
    hards = []
    retrievals = []
    for sqd, id_i in zip(D, id_data):
        _, ranked_ids = zip(*sorted(zip(sqd, id_data)))
        # 0th entry is excluded since it is always 0
        result = ranked_ids[1:] == id_i

        # soft top-k, k = 1, 2, 5, 10
        soft = [np.any(result[:k]) for k in [1, 2, 5, 10]]
        softs.append(soft)
        # hard top-k, k = 2, 3, 4
        hard = [np.all(result[:k]) for k in [2, 3, 4]]
        hards.append(hard)
        # retrieval top-k, k = 2, 3, 4
        retrieval = [np.mean(result[:k]) for k in [2, 3, 4]]
        retrievals.append(retrieval)

    average_soft = np.array(softs).mean(axis=0)
    average_hard = np.array(hards).mean(axis=0)
    average_retrieval = np.array(retrievals).mean(axis=0)
    return D, average_soft, average_hard, average_retrieval


def make_positive_pairs(num_classes, num_examples_per_class, repetition=1):
    c = num_classes
    n = num_examples_per_class
    num_pairs_per_class = n * (n - 1) / 2

    pairs_posi_class0 = np.array(list(itertools.combinations(range(n), 2)))
    offsets = n * np.repeat(np.arange(c), num_pairs_per_class)[:, None]
    pairs_posi = np.tile(pairs_posi_class0, (c, 1)) + offsets
    return np.tile(pairs_posi, (repetition, 1))


def iter_combinatorial_pairs(queue, num_examples, batch_size, interval,
                             num_classes, augment_positive=False):
    num_examples_per_class = num_examples / num_classes
    pairs = np.array(list(itertools.combinations(range(num_examples), 2)))

    if augment_positive:
        additional_positive_pairs = make_positive_pairs(
             num_classes, num_examples_per_class, num_classes - 1)
        pairs = np.concatenate((pairs, additional_positive_pairs))

    num_pairs = len(pairs)
    num_batches = num_pairs / batch_size
    perm = np.random.permutation(num_pairs)
    for i, batch_indexes in enumerate(np.array_split(perm, num_batches)):
        if i % interval == 0:
            x, c = queue.get()
            x = x.astype(np.float32) / 255.0
            c = c.ravel()
        indexes0, indexes1 = pairs[batch_indexes].T
        x0, x1, c0, c1 = x[indexes0], x[indexes1], c[indexes0], c[indexes1]
        t = np.int32(c0 == c1)  # 1 if x0 and x1 are same class, 0 otherwise
        yield x0, x1, t


class NPairMCIndexMaker(object):
    def __init__(self, batch_size, num_classes, num_per_class):
        self.batch_size = batch_size        # number of examples in a batch
        self.num_classes = num_classes      # number of classes
        self.num_per_class = num_per_class  # number of examples per class

    def get_epoch_indexes(self):
        B = self.batch_size
        K = self.num_classes
        M = self.num_per_class
        N = K * M  # number of total examples
        num_batches = M * int(K / B)  # number of batches per epoch

        indexes = np.arange(N, dtype=np.int32).reshape(K, M)
        epoch_indexes = []
        for m in range(M):
            perm = np.random.permutation(K)
            c_batches = np.array_split(perm, num_batches / M)
            for c_batch in c_batches:
                b = len(c_batch)  # actual number of examples of this batch
                indexes_anchor = M * c_batch + m

                positive_candidates = np.delete(indexes[c_batch], m, axis=1)
                indexes_positive = positive_candidates[
                    range(b), np.random.choice(M - 1, size=b)]

                epoch_indexes.append((indexes_anchor, indexes_positive))

        return epoch_indexes
