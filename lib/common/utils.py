# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:37:08 2017

@author: sakurai
"""

from collections import defaultdict
import itertools
import os
import numpy as np
import yaml

import chainer


def load_params(filename):
    with open(filename) as f:
        params = yaml.load(f)
    return params


def make_positive_pairs(num_classes, num_examples_per_class, repetition=1):
    c = num_classes
    n = num_examples_per_class
    num_pairs_per_class = n * (n - 1) // 2

    pairs_posi_class0 = np.array(list(itertools.combinations(range(n), 2)))
    offsets = n * np.repeat(np.arange(c), num_pairs_per_class)[:, None]
    pairs_posi = np.tile(pairs_posi_class0, (c, 1)) + offsets
    return np.tile(pairs_posi, (repetition, 1))


def iter_combinatorial_pairs(queue, num_examples, batch_size, interval,
                             num_classes, augment_positive=False):
    num_examples_per_class = num_examples // num_classes
    pairs = np.array(list(itertools.combinations(range(num_examples), 2)))

    if augment_positive:
        additional_positive_pairs = make_positive_pairs(
             num_classes, num_examples_per_class, num_classes - 1)
        pairs = np.concatenate((pairs, additional_positive_pairs))

    num_pairs = len(pairs)
    num_batches = num_pairs // batch_size
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
        num_batches = M * int(K // B)  # number of batches per epoch

        indexes = np.arange(N, dtype=np.int32).reshape(K, M)
        epoch_indexes = []
        for m in range(M):
            perm = np.random.permutation(K)
            c_batches = np.array_split(perm, num_batches // M)
            for c_batch in c_batches:
                b = len(c_batch)  # actual number of examples of this batch
                indexes_anchor = M * c_batch + m

                positive_candidates = np.delete(indexes[c_batch], m, axis=1)
                indexes_positive = positive_candidates[
                    range(b), np.random.choice(M - 1, size=b)]

                epoch_indexes.append((indexes_anchor, indexes_positive))

        return epoch_indexes


class Logger(defaultdict):
    def __init__(self, root_dir_path, **kwargs):
        super(Logger, self).__init__(list, kwargs)
        if not os.path.exists(root_dir_path):
            os.makedirs(root_dir_path)
        self._root_dir_path = root_dir_path

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __str__(self):
        keys = filter(lambda key: not key.startswith('_'), self)
        return ", ".join(["{}:{}".format(key, self[key]) for key in keys])

    def save(self, dir_name):
        dir_path = os.path.join(self._root_dir_path, dir_name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        others = []
        for key, value in self.items():
            if key.startswith('_'):
                continue

            if isinstance(value, (np.ndarray, list)):
                np.save(os.path.join(dir_path, key + ".npy"), value)
            elif isinstance(value, (chainer.Chain, chainer.ChainList)):
                model_path = os.path.join(dir_path, "model.npz")
                chainer.serializers.save_npz(model_path, value)
            elif isinstance(value, chainer.Optimizer):
                optimizer_path = os.path.join(dir_path, "optimizer.npz")
                chainer.serializers.save_npz(optimizer_path, value)
            else:
                others.append("{}: {}".format(key, value))

        with open(os.path.join(dir_path, "log.txt"), "a") as f:
            text = "\n".join(others) + "\n"
            f.write(text)


class UniformDistribution(object):
    def __init__(self, low, high):
        assert low <= high
        self.low = low
        self.high = high

    def rvs(self, size=None, random_state=None):
        uniform = random_state.uniform if random_state else np.random.uniform
        return uniform(self.low, self.high, size)


class LogUniformDistribution(object):
    def __init__(self, low, high):
        assert low <= high
        self.low = low
        self.high = high

    def rvs(self, size=None, random_state=None):
        uniform = random_state.uniform if random_state else np.random.uniform
        return np.exp(uniform(np.log(self.low), np.log(self.high), size))
