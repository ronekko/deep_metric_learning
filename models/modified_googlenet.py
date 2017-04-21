# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 19:42:18 2017

@author: sakurai
"""

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

from . import googlenet


class ModifiedGoogLeNet(googlenet.GoogLeNet):

    def __init__(self, out_dims=64, normalize_output=False):
        super(ModifiedGoogLeNet, self).__init__()
        # remove links and functions
        for name in filter(lambda n: n.startswith('loss'), self._children):
            self._children.remove(name)
            delattr(self, name)
        self.functions.pop('loss3_fc')
        self.functions.pop('prob')

        self.add_link('bn_fc', L.BatchNormalization(1024))
        self.add_link('fc', L.Linear(1024, out_dims))

        image_mean = np.array([123, 117, 104], dtype=np.float32)  # RGB
        self._image_mean = image_mean[None, :, None, None]
        self.normalize_output = normalize_output

    def __call__(self, x, train=False, subtract_mean=True):
        if subtract_mean:
            x = x - self._image_mean
#        h = super(ModifiedGoogLeNet, self).__call__(
#            x, layers=['pool5'], train=train)['pool5']
#        h = self.bn_fc(h, test=not train)
#        y = self.fc(h)
#        return y
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.local_response_normalization(h, n=5, k=1, alpha=1e-4/5)
        h = F.relu(self.conv2_reduce(h))
        h = F.relu(self.conv2(h))
        h = F.local_response_normalization(h, n=5, k=1, alpha=1e-4/5)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.inc3a(h)
        h = self.inc3b(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.inc4a(h)
        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)
        h = self.inc4e(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.inc5a(h)
        h = self.inc5b(h)
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.bn_fc(h, test=not train)
        y = self.fc(h)
        if self.normalize_output:
            y = F.normalize(y)
        return y

    def to_cpu(self, device=None):
        if isinstance(self._image_mean, chainer.cuda.ndarray):
            self._image_mean = self._image_mean.get()
        return super(ModifiedGoogLeNet, self).to_cpu()

    def to_gpu(self, device=None):
        self._image_mean = chainer.cuda.cupy.asarray(self._image_mean)
        return super(ModifiedGoogLeNet, self).to_gpu()
