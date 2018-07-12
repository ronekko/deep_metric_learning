# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 20:49:04 2017

@author: sakurai
"""

import colorama

from sklearn.model_selection import ParameterSampler

from lib.functions.proxy_nca_loss import proxy_nca_loss
from lib.common.utils import LogUniformDistribution, load_params
from lib.common.train_eval import train

colorama.init()


def lossfun_one_batch(model, params, batch):
    # the first half of a batch are the anchors and the latters
    # are the positive examples corresponding to each anchor
    xp = model.xp
    x_data, c_data = batch
    x_data = xp.asarray(x_data)
    # Since the class ID starts by 1, they are shifted in order to be 0-based.
    c_data = c_data.ravel() - 1

    y = model(x_data)  # y must be normalized as unit vectors
    return proxy_nca_loss(y, model.P, c_data)


if __name__ == '__main__':
    param_filename = 'proxy_nca_cars196.yaml'
    random_search_mode = True
    random_state = None
    num_runs = 10000
    save_distance_matrix = False

    if random_search_mode:
        param_distributions = dict(
            learning_rate=LogUniformDistribution(low=2e-6, high=6e-5),
            l2_weight_decay=LogUniformDistribution(low=1e-4, high=2e-2),
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
            method='proxy_nca',  # sampling method for batch construction
            comment='bs32'
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
