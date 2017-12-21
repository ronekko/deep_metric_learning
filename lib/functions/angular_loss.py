# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 17:16:53 2017

@author: sakurai
"""

import matplotlib.pyplot as plt
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable
from chainer.functions import matmul
from chainer.functions import transpose


def angular_loss(anchor, positive, negative, alpha=45, in_degree=True,
                 reduce='mean'):
    '''
    Features, y = dnn(x), must be l2 normalized.
    '''
    if in_degree:
        alpha = np.deg2rad(alpha)
    # tan(x)^2: [0, ..., pi/4, ..., pi/3] -> [0, ..., 1, ..., 3]
    # strictly increaseing convex function
    sq_tan_alpha = np.tan(alpha) ** 2
    c = (a + p) / 2
    loss = F.relu(F.batch_l2_norm_squared(a - p)
                  - 4 * sq_tan_alpha * F.batch_l2_norm_squared(n - c))
    return loss


def angular_mc_loss(f, f_p, alpha=45, in_degree=True):
    '''
    Args:
        f (chainer.Variable or xp.npdarray):
            Anchor vectors. Each vectors in f must be l2 normalized.
        f_p (chainer.Variable or xp.npdarray):
            Positive vectors. Each vectors in f must be l2 normalized.
    '''
    xp = cuda.get_array_module(f)

    if in_degree:
        alpha = np.deg2rad(alpha)
    sq_tan_alpha = np.tan(alpha) ** 2
    n_pairs = len(f)

    # first and second term of f_{a,p,n}
    term1 = 4 * sq_tan_alpha + matmul(f + f_p, transpose(f_p))
    term2 = 2 * (1 + sq_tan_alpha) * F.sum(f * f_p, axis=1, keepdims=True)
#    term2 = 2 * (1 + sq_tan_alpha) * F.batch_matmul(f, f_p, transa=True).reshape(n_pairs, 1)

    f_apn = term1 - F.broadcast_to(term2, (n_pairs, n_pairs))
    # multiply zero to diagonal components of f_apn
    mask = xp.ones_like(f_apn.data) - xp.eye(n_pairs, dtype=f.dtype)
    f_apn = f_apn * mask

    return F.average(F.logsumexp(f_apn, axis=1))


if __name__ == '__main__':
    B, D = 3, 4
#    alpha_in_degree = 10
    alpha_in_degree = 26.56505117707799  # = np.rad2deg(np.arctan(0.5))
#
#    a = Variable(np.random.randn(B, D).astype('f'))
#    p = Variable(np.random.randn(B, D).astype('f'))
#    n = Variable(np.random.randn(B, D).astype('f'))
    a = Variable(np.array([[2, 1]]).astype('f'))
    p = Variable(np.array([[1, 3]]).astype('f'))
    n = Variable(np.array([[2, 2]]).astype('f'))

    alpha = np.deg2rad(alpha_in_degree)
    step_size = 0.01
    for i in range(100):
        plt.plot(*a.data[0], 'o', label='anchor')
        plt.plot(*p.data[0], 'o', label='positive')
        plt.plot(*n.data[0], 'o', label='negative')
        plt.axes().set_aspect('equal')
        plt.axis((0, 5, 0, 5))
        plt.grid()
        plt.legend()
        plt.show()

        loss = angular_loss(a, p, n, alpha)
#        loss = F.triplet(a, p, n, 10)
        print(i, loss)
        if np.allclose(loss.data, 0):
            break

        a.cleargrad()
        p.cleargrad()
        n.cleargrad()
        loss.backward(True)
        a.data -= step_size * a.grad
        p.data -= step_size * p.grad
        n.data -= step_size * n.grad
