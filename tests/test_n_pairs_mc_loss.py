# -*- coding: utf-8 -*-
"""
Created on Tue Feb 07 19:23:18 2017

@author: sakurai
"""


import unittest

import numpy as np

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr

from deep_metric_learning.lib.functions.n_pair_mc_loss import n_pair_mc_loss


class TestNPairsMCLoss(unittest.TestCase):

    def setUp(self):
        self.f_data = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]],
                               dtype=np.float32)
        self.f_p_data = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]],
                                 dtype=np.float32)
        self.l2_reg = 0.001

    def check_forward(self, f_data, f_p_data, l2_reg):
        xp = cuda.get_array_module(f_data)
        num_examples = len(f_data)
        f = chainer.Variable(f_data)
        f_p = chainer.Variable(f_p_data)

        loss = n_pair_mc_loss(f, f_p, l2_reg)

        loss_for_each = []
        for i in range(num_examples):
            exps = []
            for j in set(range(num_examples)) - {i}:
                exp_ij = xp.exp(f_data[i].dot(f_p_data[j]) -
                                f_data[i].dot(f_p_data[i]))
                exps.append(exp_ij)
            loss_i = xp.log(1.0 + sum(exps))
            loss_for_each.append(loss_i)
        loss_expected = xp.asarray(loss_for_each, dtype=np.float32).mean()

        testing.assert_allclose(loss.data, loss_expected, atol=1e-2)

    def test_forward_cpu(self):
        self.check_forward(self.f_data, self.f_p_data, self.l2_reg)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.f_data), cuda.to_gpu(self.f_p_data), self.l2_reg)

    def check_backward(self, f, f_p, l2_reg):
        gradient_check.check_backward(
            lambda f, f_p: n_pair_mc_loss(f, f_p, l2_reg), (f, f_p),
            None, atol=1.e-1)

    def test_backward_cpu(self):
        self.check_backward(self.f_data, self.f_p_data, self.l2_reg)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.f_data), cuda.to_gpu(self.f_p_data), self.l2_reg)


testing.run_module(__name__, __file__)
