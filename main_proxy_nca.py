# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 20:49:04 2017

@author: sakurai
"""

import colorama
import numpy as np

import chainer.functions as F
from sklearn.model_selection import ParameterSampler

from lib.functions.n_pair_mc_loss import n_pair_mc_loss
from lib.common.utils import LogUniformDistribution, load_params
from lib.common.train_eval import train

colorama.init()


def squared_distance_matrix(X, Y=None):
    if Y is None:
        Y = X
    return F.sum(((X[:, None] - Y[None]) ** 2), -1)


def squared_distance_matrix2(X, Y=None):
    XX = F.batch_l2_norm_squared(X)
    if Y is None:
        Y = X
        YY = XX
    else:
        YY = F.batch_l2_norm_squared(Y)
    m = len(X)
    n = len(Y)

    distances = -2.0 * F.matmul(X, Y.T)
    distances = distances + F.broadcast_to(F.expand_dims(XX, 1), (m, n))
    distances = distances + F.broadcast_to(F.expand_dims(YY, 0), (m, n))
    # TODO: Is this necessary?
    distances = F.relu(distances)  # Force to be nonnegative
    return distances


# Implementation with softmax
def lossfun_one_batch(model, params, batch):
    # the first half of a batch are the anchors and the latters
    # are the positive examples corresponding to each anchor
    xp = model.xp
    x_data, c_data = batch
    x_data = xp.asarray(x_data)
    c_data = c_data.ravel() - 1

    y = model(x_data)  # y must be normalized as unit vectors

    # Forcely normalizing the norm of each proxy
    # TODO: Is this safe? (This operation is done out of computation graph)
#    model.P.array /= xp.linalg.norm(model.P.array, axis=1, keepdims=True)

    d = squared_distance_matrix(y, F.normalize(model.P))
    prob = F.softmax(-d)
    loss = F.log(1.0 / prob[np.arange(len(y)), c_data] - 1.0)
    return F.average(loss)


## Implementation faithful (but naive) to equation (4)
#def lossfun_one_batch(model, params, batch):
#    # the first half of a batch are the anchors and the latters
#    # are the positive examples corresponding to each anchor
#    xp = model.xp
#    x_data, c_data = batch
#    x_data = xp.asarray(x_data)
#    c_data = c_data.ravel() - 1
#
#    y = model(x_data)  # y must be normalized as unit vectors
#
#    # Forcely normalizing the norm of each proxy
#    # TODO: Is this safe? (This operation is done out of computation graph)
##    model.P.array /= xp.linalg.norm(model.P.array, axis=1, keepdims=True)
#
#    proxy = model.P
#    d = squared_distance_matrix(y, F.normalize(proxy))
#    inv_exp = F.exp(-d)
#    positive = inv_exp[np.arange(len(y)), c_data]
#    loss = -F.log(positive / (F.sum(inv_exp, axis=1) - positive))
#    return F.average(loss)


## Implementation faithful (with logsumexp) to equation (4)
#def lossfun_one_batch(model, params, batch):
#    # the first half of a batch are the anchors and the latters
#    # are the positive examples corresponding to each anchor
#    xp = model.xp
#    x_data, c_data = batch
#    x_data = xp.asarray(x_data)
#    c_data = c_data.ravel() - 1
#
#    y = model(x_data)  # y must be normalized as unit vectors
#
#    # Forcely normalizing the norm of each proxy
#    # TODO: Is this safe? (This operation is done out of computation graph)
##    model.P.array /= xp.linalg.norm(model.P.array, axis=1, keepdims=True)
#
#    proxy = model.P
#    distance = squared_distance_matrix(y, F.normalize(proxy))
#
#    d_posi = distance[np.arange(len(y)), c_data]
#
#    B, K = distance.shape  # batch size and the number of classes
#    # For each row, remove one element corresponding to the positive distance
#    mask = np.tile(np.arange(K), (B, 1)) != c_data[:, None]
#    d_nega = distance[mask].reshape(B, K - 1)
#
#    log_denominator = F.logsumexp(-d_nega, axis=1)
#    loss = d_posi + log_denominator
#    return F.average(loss)


if __name__ == '__main__':
    param_filename = 'proxy_nca_cars196.yaml'
    random_search_mode = True
    random_state = None
    num_runs = 10000
    save_distance_matrix = False

    if random_search_mode:
        param_distributions = dict(
            learning_rate=LogUniformDistribution(low=2e-6, high=2e-4),
            l2_weight_decay=LogUniformDistribution(low=1e-5, high=1e-2),
#            out_dim=[64, 128, 256],
            optimizer=['RMSprop', 'Adam']  # 'RMSPeop' or 'Adam'
        )
        static_params = dict(
            num_epochs=15,
            num_batches_per_epoch=1875,
            batch_size=32,
            out_dim=256,
#            learning_rate=7e-5,
            crop_size=224,
            normalize_output=True,
#            l2_weight_decay=5e-3,
#            optimizer='Adam',  # 'Adam' or 'RMSPeop'
            distance_type='cosine',  # 'euclidean' or 'cosine'
            dataset='cars196',  # 'cars196' or 'cub200_2011' or 'products'
            method='clustering',  # sampling method for batch construction
            comment='softmax'
        )

        sampler = ParameterSampler(param_distributions, num_runs, random_state)

        for random_params in sampler:
            params = {}
            params.update(random_params)
            params.update(static_params)

            stop = train(__file__, lossfun_one_batch, params,
                         save_distance_matrix)
            if stop:
                break
    else:
        print('Train once using config file "{}".'.format(param_filename))
        params = load_params(param_filename)
        train(__file__, lossfun_one_batch, params, save_distance_matrix)
