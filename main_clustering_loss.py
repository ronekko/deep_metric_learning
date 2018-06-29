# -*- coding: utf-8 -*-
"""
Created on Mon Feb 06 17:12:58 2017

@author: sakurai
"""

import colorama

from sklearn.model_selection import ParameterSampler

from lib.functions.clustering_loss import clustering_loss
from lib.common.utils import (
    UniformDistribution, LogUniformDistribution, load_params)
from lib.common.train_eval import train

colorama.init()


def lossfun_one_batch(model, params, batch):
    x_data, c_data = batch
    x_data = model.xp.asarray(x_data)
    c_data = model.xp.asarray(c_data)

    y = model(x_data)

    # decay gamma at regular interval
    if type(params.gamma) is not float:
        params.gamma = params.gamma_init
        params.num_updates = 0
    else:
        if (params.num_updates != 0 and
                params.num_updates % params.num_batches_per_epoch == 0):
            params.gamma *= params.gamma_decay
        params.num_updates += 1

    return clustering_loss(y, c_data, params.gamma)


if __name__ == '__main__':
    param_filename = 'clustering_cub200_2011.yaml'
    random_search_mode = True
    random_state = None
    num_runs = 100000
    save_distance_matrix = False

    if random_search_mode:
        param_distributions = dict(
            learning_rate=LogUniformDistribution(low=1e-6, high=1e-4),
            gamma_init=LogUniformDistribution(low=1e+1, high=1e+4),
            gamma_decay=UniformDistribution(low=0.7, high=1.0),
            l2_weight_decay=LogUniformDistribution(low=1e-5, high=1e-2),
            optimizer=['RMSProp', 'Adam']  # 'RMSPeop' or 'Adam'
        )
        static_params = dict(
            num_epochs=15,
            num_batches_per_epoch=500,
            batch_size=120,
            out_dim=64,
#            learning_rate=0.0001,
#            gamma_init=10.0,
#            gamma_decay=0.94,
            crop_size=224,
            normalize_output=True,
#            l2_weight_decay=0,  # non-negative constant
#            optimizer='RMSProp',  # 'Adam' or 'RMSPeop'
            distance_type='euclidean',  # 'euclidean' or 'cosine'
            dataset='cub200_2011',  # 'cars196' or 'cub200_2011' or 'products'
            method='clustering'  # sampling method for batch construction
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
