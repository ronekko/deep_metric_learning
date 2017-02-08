# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 19:42:25 2017

@author: sakurai
"""

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.optimizers
from chainer.dataset.convert import concat_examples

import chainer_datasets
import googlenet


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

        image_mean = np.array([123, 117, 104], dtype=np.float32) / 225.0  # RGB
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


class SimpleConvnet(chainer.Chain):
    def __init__(self, out_dim):
        super(SimpleConvnet, self).__init__(
            conv1=L.Convolution2D(3, 50, 3),
            bn_conv1=L.BatchNormalization(50),
            conv21=L.Convolution2D(50, 100, 3),
            bn_conv21=L.BatchNormalization(100),
            conv22=L.Convolution2D(100, 100, 1),
            bn_conv22=L.BatchNormalization(100),
            conv31=L.Convolution2D(100, 200, 3),
            bn_conv31=L.BatchNormalization(200),
            conv32=L.Convolution2D(200, 200, 3),
            bn_conv32=L.BatchNormalization(200),
            conv41=L.Convolution2D(200, 400, 3),
            bn_conv41=L.BatchNormalization(400),
            conv42=L.Convolution2D(400, 400, 1),
            bn_conv42=L.BatchNormalization(400),
            conv5=L.Convolution2D(400, 400, 1),
            bn_conv5=L.BatchNormalization(400),
            conv6=L.Convolution2D(400, 400, 1),
            bn_conv6=L.BatchNormalization(400),
            linear1=L.Linear(400, 400),
            bn_linear1=L.BatchNormalization(400),
            linear2=L.Linear(400, out_dim)
        )

    def __call__(self, x, train=False):
        h = self.conv1(x)
        h = self.bn_conv1(h, test=not train)
        h = F.max_pooling_2d(h, 2)
        h = F.relu(h)

        h = self.conv21(h)
        h = self.bn_conv21(h, test=not train)
        h = F.relu(h)
        h = self.conv22(h)
        h = self.bn_conv22(h, test=not train)
        h = F.max_pooling_2d(h, 2)
        h = F.relu(h)

        h = self.conv31(h)
        h = self.bn_conv31(h, test=not train)
        h = F.relu(h)
        h = self.conv32(h)
        h = self.bn_conv32(h, test=not train)
        h = F.max_pooling_2d(h, 2)
        h = F.relu(h)

        h = self.conv41(h)
        h = self.bn_conv41(h, test=not train)
        h = F.relu(h)
        h = self.conv42(h)
        h = self.bn_conv42(h, test=not train)
        h = F.max_pooling_2d(h, 2)
        h = F.relu(h)

        h = self.conv5(h)
        h = self.bn_conv5(h, test=not train)
        h = F.relu(h)

        h = self.conv6(h)
        h = self.bn_conv6(h, test=not train)
        h = F.relu(h)

        h = F.average_pooling_2d(h, h.shape[2:])
        h = self.linear1(h)
        h = self.bn_linear1(h, test=not train)
#        h = F.dropout(h, ratio=0.5, train=train)
        h = F.relu(h)
        h = self.linear2(h)
        return h


if __name__ == '__main__':
    batch_size = 60

    # load database
    iters = chainer_datasets.get_iterators(batch_size)
    iter_train, iter_train_eval, iter_test = iters

    # load model
    model = ModifiedGoogLeNet().to_gpu()
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    x, c = concat_examples(next(iter_train), device=0)
    y = model(x, train=True)
    loss = sum(sum(y))
    optimizer.zero_grads()
    loss.backward()
