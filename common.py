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


def iterate_forward(model, epoch_iterator, train=False, normalize=False):
    xp = model.xp
    y_batches = []
    c_batches = []
    for batch in copy.copy(epoch_iterator):
        x_batch_data, c_batch_data = batch
        x_batch = Variable(xp.asarray(x_batch_data), volatile=not train)
        y_batch = model(x_batch, train=train)
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


# average accuracy and distance matrix for test data
def evaluate(model, epoch_iterator, distance='euclidean', normalize=False):
    # fprop to calculate distance matrix (not for backprop)
    y_data, c_data = iterate_forward(
        model, epoch_iterator, train=False, normalize=normalize)

    # compute the distance matrix of the list of ys
    if distance == 'euclidean':
        D = euclidean_distances(y_data, squared=False)
    elif distance == 'cosine':
        D = cosine_distances(y_data)
    else:
        raise ValueError("distance must be 'euclidean' or 'cosine'.")

    soft, hard, retrieval = compute_soft_hard_retrieval(D, c_data)
    return D, soft, hard, retrieval


def compute_soft_hard_retrieval(distance_matrix, labels):
    softs = []
    hards = []
    retrievals = []
    for d_i, label_i in zip(distance_matrix, labels):
        ranked_labels = labels[np.argsort(d_i)]
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


class Logger(defaultdict):
    def __init__(self, root_dir_path):
        super(Logger, self).__init__(list)
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
        for key, value in self.iteritems():
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


def normalized_mutual_info_score(x, y):
    xp = chainer.cuda.get_array_module(x)

    contingency = contingency_matrix(x, y)
    nonzero_mask = contingency != 0
    nonzero_val = contingency[nonzero_mask]

    pi = contingency.sum(axis=1, keepdims=True)
    pj = contingency.sum(axis=0, keepdims=True)
    total_mass = pj.sum()
    pi /= total_mass
    pj /= total_mass
    pi_pj = (pj * pi)[nonzero_mask]

    pij = nonzero_val / total_mass
    log_pij = xp.log(pij)
    log_pi_pj = xp.log(pi_pj)
    mi = xp.sum(pij * (log_pij - log_pi_pj))
    nmi = mi / max(xp.sqrt(entropy(pi) * entropy(pj)), 1e-10)
    return xp.clip(nmi, 0, 1)


def contingency_matrix(x, y):
    xp = chainer.cuda.get_array_module(x)
    n_bins_x = int(x.max()) + 1
    n_bins_y = int(y.max()) + 1
    n_bins = n_bins_x * n_bins_y
    i = x * n_bins_y + y
    flat_contingency = xp.bincount(i, xp.ones(len(i)), minlength=n_bins)
    return flat_contingency.reshape((n_bins_x, n_bins_y))


def entropy(p):
    if len(p) == 0:
        return 1.0
    xp = chainer.cuda.get_array_module(p)
    p = p[p > 0]
    return xp.sum(p * xp.log(p))
