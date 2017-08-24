# -*- coding: utf-8 -*-
"""
Created on Tue Feb 07 16:37:21 2017

@author: sakurai
"""

import unittest

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr

from deep_metric_learning.lib.functions.clustering_loss import clustering_loss


class TestClusteringLoss(unittest.TestCase):

    def setUp(self):
        self.x_data = np.array([-11, -10, -8, 0, 1, 2, 9, 10, 11],
                               dtype=np.float32).reshape(-1, 1)
        self.c_data = np.array([1, 1, 1, 2, 2, 2, 0, 0, 0])  # class labels
        self.gamma = 10.0
        self.T = 5
        self.y_star = np.array([1, 1, 1, 4, 4, 4, 7, 7, 7])
        self.y_pam = np.array([1, 1, 1, 4, 4, 4, 7, 7, 7])

    def check_forward(self, x_data, c_data, gamma, T, y_star, y_pam):
        num_examples = len(x_data)
        x = chainer.Variable(x_data)
        c = chainer.Variable(c_data)

        loss = clustering_loss(x, c, gamma, T)

        sq_distances_ij = []
        for i, j in zip(range(num_examples), y_pam):
            sqd_ij = np.sum((x_data[i] - x_data[j]) ** 2)
            sq_distances_ij.append(sqd_ij)
        f = -sum(sq_distances_ij)

        sq_distances_ij = []
        for i, j in zip(range(num_examples), y_star):
            sqd_ij = np.sum((x_data[i] - x_data[j]) ** 2)
            sq_distances_ij.append(sqd_ij)
        f_tilde = -sum(sq_distances_ij)

        delta = 1.0 - normalized_mutual_info_score(cuda.to_cpu(c_data), y_pam)
        loss_expected = f + gamma * delta - f_tilde

        testing.assert_allclose(loss.data, loss_expected)

    def test_forward_cpu(self):
        self.check_forward(self.x_data, self.c_data,
                           self.gamma, self.T, self.y_star, self.y_pam)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x_data), cuda.to_gpu(self.c_data),
                           self.gamma, self.T, self.y_star, self.y_pam)

    def check_backward(self, x_data, c_data, gamma, T):
        gradient_check.check_backward(
            lambda x, c: clustering_loss(x, c, gamma, T), (x_data, c_data),
            None, atol=1.e-1)

    def test_backward_cpu(self):
        self.check_backward(self.x_data, self.c_data, self.gamma, self.T)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x_data), cuda.to_gpu(self.c_data),
            self.gamma, self.T)


testing.run_module(__name__, __file__)
