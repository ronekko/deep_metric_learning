# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 20:49:04 2017

@author: sakurai
"""

import colorama

import chainer.functions as F
from sklearn.model_selection import ParameterSampler

from lib.functions.n_pair_mc_loss import n_pair_mc_loss
from lib.common.utils import LogUniformDistribution, load_params
from lib.common.train_eval import train

colorama.init()


def lossfun_one_batch(model, params, batch):
    # the first half of a batch are the anchors and the latters
    # are the positive examples corresponding to each anchor
    x_data, c_data = batch
    x_data = model.xp.asarray(x_data)

    y = model(x_data)
    y_a, y_p = F.split_axis(y, 2, axis=0)

    return n_pair_mc_loss(y_a, y_p, params.loss_l2_reg)


if __name__ == '__main__':
    param_filename = 'n_pair_mc_cub200_2011.yaml'
    random_search_mode = True
    random_state = None
    num_runs = 10000
    save_distance_matrix = False

    if random_search_mode:
        param_distributions = dict(
            learning_rate=LogUniformDistribution(low=3e-5, high=3e-4),
            loss_l2_reg=LogUniformDistribution(low=1e-6, high=5e-3),
            l2_weight_decay=LogUniformDistribution(low=1e-5, high=1e-2),
            out_dim=[64, 128, 256, 512],
#            optimizer=['RMSProp', 'Adam']  # 'RMSPeop' or 'Adam'
        )
        static_params = dict(
            num_epochs=8,
            num_batches_per_epoch=500,
            batch_size=120,
#            out_dim=64,
#            learning_rate=7.10655234311e-05,
#            loss_l2_reg=2.80690151536e-06,  # L2-norm penalty for output vector
            crop_size=224,
            normalize_output=False,
#            l2_weight_decay=0.00579416451873,
            optimizer='Adam',  # 'Adam' or 'RMSPeop'
            distance_type='euclidean',  # 'euclidean' or 'cosine'
            dataset='cars196',  # 'cars196' or 'cub200_2011' or 'products'
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
