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
import chainer.functions as F

from deep_metric_learning.lib.functions.proxy_nca_loss import proxy_nca_loss


class TestProxyNcaLoss(unittest.TestCase):

    def setUp(self):
        batch_size = 5
        n_classes = 10
        out_dims = 3
        self.x_data = np.random.randn(batch_size, out_dims).astype(np.float32)
        # x_data is assumed that each vector is L2 normalized
        self.x_data /= np.linalg.norm(self.x_data, axis=1, keepdims=True)
        self.proxy_data = np.random.randn(
            n_classes, out_dims).astype(np.float32)
        self.labels_data = np.random.choice(n_classes, batch_size)

    def check_forward(self, x_data, proxy_data, labels_data):
        x = chainer.Variable(x_data)
        proxy = chainer.Variable(proxy_data)

        x = F.normalize(x)
        loss = proxy_nca_loss(x, proxy, labels_data)
        self.assertEqual(loss.dtype, np.float32)

    def test_forward_cpu(self):
        self.check_forward(self.x_data, self.proxy_data, self.labels_data)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x_data),
                           cuda.to_gpu(self.proxy_data),
                           self.labels_data)

    def check_backward(self, x_data, proxy_data, labels_data):
        gradient_check.check_backward(
            lambda x, p: proxy_nca_loss(x, p, labels_data),
            (x_data, proxy_data), None, atol=1.e-1)

    def test_backward_cpu(self):
        self.check_backward(self.x_data, self.proxy_data, self.labels_data)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x_data),
                            cuda.to_gpu(self.proxy_data),
                            self.labels_data)


testing.run_module(__name__, __file__)
