# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 01:55:58 2016

@author: sakurai
"""

from collections import defaultdict
import copy
import itertools
import os
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

import chainer
from chainer import cuda
from chainer import Variable
from chainer import Optimizer
from chainer import Chain, ChainList
from chainer.serializers import save_npz
from tqdm import tqdm


def iterate_forward(model, epoch_iterator, normalize=False):
    xp = model.xp
    y_batches = []
    c_batches = []
    for batch in tqdm(copy.copy(epoch_iterator)):
        x_batch_data, c_batch_data = batch
        x_batch = Variable(xp.asarray(x_batch_data))
        y_batch = model(x_batch)
        if normalize:
            y_batch_data = y_batch.data / xp.linalg.norm(
                y_batch.data, axis=1, keepdims=True)
        else:
            y_batch_data = y_batch.data
        y_batches.append(y_batch_data)
        y_batch = None
        c_batches.append(c_batch_data)
    y_data = cuda.to_cpu(xp.concatenate(y_batches))
    c_data = np.concatenate(c_batches)
    return y_data, c_data


# memory friendly average accuracy for test data
def evaluate(model, epoch_iterator, distance='euclidean', normalize=False,
             batch_size=10, return_distance_matrix=False):
    if distance not in ('cosine', 'euclidean'):
        raise ValueError("distance must be 'euclidean' or 'cosine'.")

    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            y_data, c_data = iterate_forward(
                    model, epoch_iterator, normalize=normalize)

    add_epsilon = True
    xp = cuda.get_array_module(y_data)
    num_examples = len(y_data)

    D_batches = []
    softs = []
    hards = []
    retrievals = []
    yy = xp.sum(y_data ** 2.0, axis=1)

    if distance == 'cosine':
        y_data = y_data / yy[:, None]  # L2 normalization

    for start in range(0, num_examples, batch_size):
        end = start + batch_size
        if end > num_examples:
            end = num_examples
        y_batch = y_data[start:end]
        yy_batch = yy[start:end]
        c_batch = c_data[start:end]

        D_batch = yy + yy_batch[:, None] - 2.0 * xp.dot(y_batch, y_data.T)
        xp.maximum(D_batch, 0, out=D_batch)
        if add_epsilon:
            D_batch += 1e-40
        # ensure the diagonal components are zero
        xp.fill_diagonal(D_batch[:, start:end], 0)

        soft, hard, retr = compute_soft_hard_retrieval(
            D_batch, c_data, c_batch)

        softs.append(len(y_batch) * soft)
        hards.append(len(y_batch) * hard)
        retrievals.append(len(y_batch) * retr)
        if return_distance_matrix:
            D_batches.append(D_batch)

    avg_softs = xp.sum(softs, axis=0) / num_examples
    avg_hards = xp.sum(hards, axis=0) / num_examples
    avg_retrievals = xp.sum(retrievals, axis=0) / num_examples

    if return_distance_matrix:
        D = cuda.to_cpu(xp.vstack(D_batches))
    else:
        D = None
    return D, avg_softs, avg_hards, avg_retrievals


def compute_soft_hard_retrieval(distance_matrix, labels, label_batch=None):
    softs = []
    hards = []
    retrievals = []

    if label_batch is None:
        label_batch = labels
    distance_matrix = cuda.to_cpu(distance_matrix)
    labels = cuda.to_cpu(labels)
    label_batch = cuda.to_cpu(label_batch)

    K = 11  # "K" for top-K
    for d_i, label_i in zip(distance_matrix, label_batch):
        top_k_indexes = np.argpartition(d_i, K)[:K]
        sorted_top_k_indexes = top_k_indexes[np.argsort(d_i[top_k_indexes])]
        ranked_labels = labels[sorted_top_k_indexes]
        # 0th entry is excluded since it is always 0
        ranked_hits = ranked_labels[1:] == label_i

        # soft top-k, k = 1, 2, 5, 10
        soft = [np.any(ranked_hits[:k]) for k in [1, 2, 5, 10]]
        softs.append(soft)
        # hard top-k, k = 2, 3, 4
        hard = [np.all(ranked_hits[:k]) for k in [2, 3, 4]]
        hards.append(hard)
        # retrieval top-k, k = 2, 3, 4
        retrieval = [np.mean(ranked_hits[:k]) for k in [2, 3, 4]]
        retrievals.append(retrieval)

    average_soft = np.array(softs).mean(axis=0)
    average_hard = np.array(hards).mean(axis=0)
    average_retrieval = np.array(retrievals).mean(axis=0)
    return average_soft, average_hard, average_retrieval


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
            elif isinstance(value, (Chain, ChainList)):
                model_path = os.path.join(dir_path, "model.npz")
                save_npz(model_path, value)
            elif isinstance(value, Optimizer):
                optimizer_path = os.path.join(dir_path, "optimizer.npz")
                save_npz(optimizer_path, value)
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
