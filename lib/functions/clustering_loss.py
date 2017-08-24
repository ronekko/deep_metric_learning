# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:19:20 2017

@author: sakurai
"""

import copy
import numpy as np
import contexttimer

import chainer
from chainer import cuda
import chainer.functions as F


def clustering_loss(x, t, gamma, T=5):
    """Clustering loss function for metric learning.

    Args:
        x (~chainer.Variable):
            Feature vectors.
        t (~chainer.Variable):
            Class labels corresponding to x.
        gamma (~float):
            Hyperparameter gamma.
        T (int):
            Maximum number of iterations in Algorithm 2.

    Returns:
        ~chainer.Variable: Loss value.

    See: `Learnable Structured Clustering Framework for Deep Metric Learning \
         <https://arxiv.org/abs/1612.01213>`_

    """
    if not isinstance(x, chainer.Variable):
        x = chainer.Variable(x)
    if not isinstance(t, chainer.Variable):
        t = chainer.Variable(t)
    t_cpu = chainer.cuda.to_cpu(t.data).ravel()

    batch_size = len(t.data)
    num_classes = len(np.unique(t_cpu))

    v = list(range(batch_size))
    s = []

    # First, search the sub-optimal solution y_PAM of the clustering.
    # Note that this computation is done outside the computational graph.
    # Find an initial medoids of S_PAM by Algorithm 1 in the paper.
    D = distance_matrix(x.data)
    D = cuda.to_cpu(D)
    for _ in range(num_classes):
        # find an element in v which maximise a_function
        a_best = -np.inf
        for i in v:
            distances = D[s + [i]]
            g_s = distances.argmin(axis=0)
            f = -distances[g_s, range(batch_size)].sum()
            if f + gamma < a_best:  # skip if this is hopeless to be the best
                continue
            delta = 1.0 - normalized_mutual_info_score(t_cpu, g_s)
            a = f + gamma * delta
            if a > a_best:
                a_best = a
                i_best = i

        s.append(i_best)
        v.remove(i_best)

        # In order to speed-up by making skip to calculate NMI more frequently,
        # sort v in descending order by distances to their nearest medoid
        D_min = D[s].min(0)  # distance to the nearest medoid for each point
        sorted_order = np.argsort(D_min[v])[::-1]
        v = np.array(v)[sorted_order].tolist()

    # Refine S_PAM by Algorithm 2
    a_previous = a_best
    for t in range(T):
        np.random.shuffle(s)
        y_pam = np.array(s)[D[s].argmin(axis=0)]
        # since a column of D may have multiple zeros due to numerical errors,
        # ensure y_pam[j] == j, for each j \in s
        y_pam[s] = s
        for k in copy.copy(s):
            js = np.argwhere(y_pam == k).ravel()
            if len(js) == 1:
                continue
            D_k = D[:, js][js]
            fs = -D_k.sum(axis=1)
            j_max = fs.argmax()
            f_max = fs[j_max]
            s_except_k = copy.copy(s)
            s_except_k.remove(k)
            a_best = -np.inf
            for j, f in zip(js, fs):
                if f + gamma < f_max:
                    continue
                g_s_j = D[s_except_k + [j]].argmin(axis=0)
                delta = 1.0 - normalized_mutual_info_score(t_cpu, g_s_j)
                a = f + gamma * delta
                if a > a_best:
                    a_best = a
                    j_best = j
            s = s_except_k + [j_best]

        # stop if the score did not improve from the previous step
        distances = D[s]
        g_s = distances.argmin(axis=0)
        f = -distances[g_s, range(batch_size)].sum()
        delta = 1.0 - normalized_mutual_info_score(t_cpu, g_s)
        a = f + gamma * delta
        if a == a_previous:
            break
        a_previous = a
    s_pam = s

    # Here, compute the loss with S_PAM and its corresponding delta.
    y_pam = np.asarray(s_pam)[D[s_pam].argmin(axis=0)].tolist()

    y_star = np.empty_like(t_cpu)
    for c in np.unique(t_cpu):
        js = np.argwhere(t_cpu == c).ravel()  # indexes of examples of class c
        D_c = D[:, js][js]
        fs = D_c.sum(axis=1)
        y_star_c = js[fs.argmin()]
        y_star[js] = y_star_c

    f = -F.sum(F.batch_l2_norm_squared(x - x[y_pam]))
    f_tilde = -F.sum(F.batch_l2_norm_squared(x - x[y_star]))
    loss = F.relu(f + gamma * delta - f_tilde)
    return loss


def distance_matrix(x, add_epsilon=True):
    xp = chainer.cuda.get_array_module(x)
    xx = xp.sum(x ** 2.0, axis=1)
    mat = xx + xx[:, None] - 2.0 * xp.dot(x, x.T)
    xp.maximum(mat, 0, out=mat)
    if add_epsilon:
        mat += 1e-40
    # ensure the diagonal components are zero
    xp.fill_diagonal(mat, 0)
    return mat


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


if __name__ == '__main__':
#    x_train = np.load('y_train.npy')
#    c_train = np.load('c_train.npy').ravel()
    x_test = np.load('y_test.npy')
    c_test = np.load('c_test.npy').ravel()

    N, D = x_test.shape
    M = 128  # number of examples in a minibatch
    K = 99  # number of classes
    gamma = 1.0

#    np.random.seed(0)
    _i = np.random.choice(N, M, False)
    _x = x_test[_i]
    _c = c_test[_i]

    with contexttimer.Timer() as timer:
        loss = clustering_loss(_x, _c, gamma)
    print(timer.elapsed, ' [s]')
    print('loss:', loss.data)
