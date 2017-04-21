# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 19:44:37 2017

@author: sakurai
"""

import chainer
import chainer.functions as F
import chainer.links as L


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
