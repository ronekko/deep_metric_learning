# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 20:49:04 2017

@author: sakurai
"""

import colorama

import chainer.functions as F
from sklearn.model_selection import ParameterSampler

from lib.functions.lifted_struct_loss import lifted_struct_loss
from lib.common.utils import LogUniformDistribution, load_params
from lib.common.train_eval import train

colorama.init()


def lossfun_one_batch(model, params, batch):
    # the first half of a batch are the anchors and the latters
    # are the positive examples corresponding to each anchor
    x_data, c_data = batch
    x_data = model.xp.asarray(x_data)
    c_data = model.xp.asarray(c_data)

    y = model(x_data)
    y_a, y_p = F.split_axis(y, 2, axis=0)

    return lifted_struct_loss(y_a, y_p, params.alpha)


if __name__ == '__main__':
    param_filename = 'lifted_struct_cub200_2011.yaml'
    random_search_mode = True
    random_state = None
    num_runs = 100
    save_distance_matrix = False

    if random_search_mode:
        param_distributions = dict(
            learning_rate=LogUniformDistribution(low=1e-4, high=1e-4),
#            l2_weight_decay=LogUniformDistribution(low=1e-4, high=1e-3),
#            optimizer=['RMSProp', 'Adam']  # 'RMSPeop' or 'Adam'
        )
        static_params = dict(
            num_epochs=40,
            num_batches_per_epoch=500,
            batch_size=120,
            out_dim=64,
#            learning_rate=0.0001,
            alpha=1.0,  # penalty for the norm of the output vector
            crop_size=224,
            normalize_output=False,
            l2_weight_decay=0.001,  # non-negative constant
            optimizer='RMSProp',  # 'Adam' or 'RMSPeop'
            distance_type='euclidean',  # 'euclidean' or 'cosine'
            dataset='cub200_2011',  # 'cars196' or 'cub200_2011' or 'products'
            method='n_pairs_mc'  # sampling method for batch construction
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
