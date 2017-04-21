# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:40:26 2017

@author: sakurai
"""

import numpy as np
import chainer.functions as F


def squared_distance_matrix(X):
    n = X.shape[0]
    XX = F.sum(X ** 2.0, axis=1)
    distances = -2.0 * F.linear(X, X)
    distances = distances + F.broadcast_to(XX, (n, n))
    distances = distances + F.broadcast_to(F.expand_dims(XX, 1), (n, n))
    return distances


def lifted_struct_loss(f_a, f_p, alpha=1.0):
    """Lifted struct loss function.

    Args:
        f_a (~chainer.Variable): Feature vectors as anchor examples.
            All examples must be different classes each other.
        f_p (~chainer.Variable): Positive examples corresponding to f_a.
            Each example must be the same class for each example in f_a.
        alpha (~float): The margin parameter.

    Returns:
        ~chainer.Variable: Loss value.

    See: `Deep Metric Learning via Lifted Structured Feature Embedding \
        <http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/\
        Song_Deep_Metric_Learning_CVPR_2016_paper.pdf>`_

    """
    assert f_a.shape == f_p.shape, 'f_a and f_p must have same shape.'
    n = 2 * f_a.shape[0]  # use shape[0] due to len(Variable) returns its size
    f = F.vstack((f_a, f_p))
    D_sq = squared_distance_matrix(f)

    pairs_p = np.arange(n).reshape(2, -1)  # indexes of positive pairs
    row = []
    col = []
    for i, j in pairs_p.T:
        row.append([i] * (n - 2) + [j] * (n - 2))
        col.append(np.tile(np.delete(np.arange(n), (i, j)), 2))
    row = np.ravel(row)
    col = np.ravel(col)
    pairs_n = np.vstack((row, col))

    distances_n = F.reshape(distances_n, (n // 2, -1))
    distances_p = F.sqrt(D_sq[pairs_p[0], pairs_p[1]])
    distances_n = F.sqrt(D_sq[pairs_n[0], pairs_n[1]])
    loss_ij = F.logsumexp(alpha - distances_n, axis=1) + distances_p
    return F.sum(F.relu(loss_ij) ** 2) / n


if __name__ == '__main__':
    N = 120
    D = 64
    alpha = 1.0

    f_a = np.random.randn(N // 2, D).astype(np.float32)
    f_p = np.random.randn(N // 2, D).astype(np.float32)

    import chainer
    f_a_cpu = chainer.Variable(f_a)
    f_p_cpu = chainer.Variable(f_p)
    loss_cpu = lifted_struct_loss(f_a_cpu, f_p_cpu, alpha)
    loss_cpu.backward()
    ########
    import cupy
    f_a = cupy.asarray(f_a)
    f_p = cupy.asarray(f_p)
    f_a_gpu = chainer.Variable(f_a)
    f_p_gpu = chainer.Variable(f_p)
    ########

    loss_gpu = lifted_struct_loss(f_a_gpu, f_p_gpu, alpha)
    loss_gpu.backward()

    assert np.allclose(f_a_cpu.grad, f_a_gpu.grad.get())
    assert np.allclose(f_p_cpu.grad, f_p_gpu.grad.get(), rtol=1e-3)
    assert np.allclose(loss_cpu.grad, loss_gpu.grad.get())
