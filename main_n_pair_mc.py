# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 20:49:04 2017

@author: sakurai
"""

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import six

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
from tqdm import tqdm
import colorama
from sklearn.model_selection import ParameterSampler

from functions.n_pair_mc_loss import n_pair_mc_loss
import common
from datasets import data_provider
from models.modified_googlenet import ModifiedGoogLeNet
from common import LogUniformDistribution

colorama.init()


def main(param_dict):
    script_filename = os.path.splitext(os.path.basename(__file__))[0]
    device = 0
    xp = chainer.cuda.cupy
    config_parser = six.moves.configparser.ConfigParser()
    config_parser.read('config')
    log_dir_path = os.path.expanduser(config_parser.get('logs', 'dir_path'))

    p = common.Logger(log_dir_path, **param_dict)  # hyperparameters

    ##########################################################
    # load database
    ##########################################################
    streams = data_provider.get_streams(p.batch_size, dataset=p.dataset)
    stream_train, stream_train_eval, stream_test = streams
    iter_train = stream_train.get_epoch_iterator()

    ##########################################################
    # construct the model
    ##########################################################
    model = ModifiedGoogLeNet(p.out_dim, p.normalize_output)
    if device >= 0:
        model.to_gpu()
    xp = model.xp
    if p.optimizer == 'Adam':
        optimizer = optimizers.Adam(p.learning_rate)
    elif p.optimizer == 'RMSProp':
        optimizer = optimizers.RMSprop(p.learning_rate)
    else:
        raise ValueError
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(p.l2_weight_decay))

    logger = common.Logger(log_dir_path)
    logger.soft_test_best = [0]
    time_origin = time.time()
    try:
        for epoch in range(p.num_epochs):
            time_begin = time.time()
            epoch_losses = []

            for i in tqdm(range(p.num_batches_per_epoch),
                          desc='# {}'.format(epoch)):
                # the first half of a batch are the anchors and the latters
                # are the positive examples corresponding to each anchor
                x_data, c_data = next(iter_train)
                if device >= 0:
                    x_data = cuda.to_gpu(x_data, device)
                    c_data = cuda.to_gpu(c_data, device)
                y = model(x_data, train=True)
                y_a, y_p = F.split_axis(y, 2, axis=0)

                loss = n_pair_mc_loss(y_a, y_p, p.loss_l2_reg)
                optimizer.zero_grads()
                loss.backward()
                optimizer.update()

                epoch_losses.append(loss.data)
                y = y_a = y_p = loss = None

            loss_average = cuda.to_cpu(xp.array(
                xp.hstack(epoch_losses).mean()))

            # average accuracy and distance matrix for training data
            D, soft, hard, retrieval = common.evaluate(
                model, stream_train_eval.get_epoch_iterator(), p.distance_type)

            # average accuracy and distance matrix for testing data
            D_test, soft_test, hard_test, retrieval_test = common.evaluate(
                model, stream_test.get_epoch_iterator(), p.distance_type)

            time_end = time.time()
            epoch_time = time_end - time_begin
            total_time = time_end - time_origin

            logger.epoch = epoch
            logger.total_time = total_time
            logger.loss_log.append(loss_average)
            logger.train_log.append([soft[0], hard[0], retrieval[0]])
            logger.test_log.append(
                [soft_test[0], hard_test[0], retrieval_test[0]])

            # retain the model if it scored the best test acc. ever
            if soft_test[0] > logger.soft_test_best[0]:
                logger.model_best = copy.deepcopy(model)
                logger.optimizer_best = copy.deepcopy(optimizer)
                logger.epoch_best = epoch
                logger.D_best = D
                logger.D_test_best = D_test
                logger.soft_best = soft
                logger.soft_test_best = soft_test
                logger.hard_best = hard
                logger.hard_test_best = hard_test
                logger.retrieval_best = retrieval
                logger.retrieval_test_best = retrieval_test

            print("#", epoch)
            print("time: {} ({})".format(epoch_time, total_time))
            print("[train] loss:", loss_average)
            print("[train] soft:", soft)
            print("[train] hard:", hard)
            print("[train] retr:", retrieval)
            print("[test]  soft:", soft_test)
            print("[test]  hard:", hard_test)
            print("[test]  retr:", retrieval_test)
            print("[best]  soft: {} (at # {})".format(logger.soft_test_best,
                                                      logger.epoch_best))
            print(p)
            # print norms of the weights
            params = xp.hstack([xp.linalg.norm(param.data)
                                for param in model.params()]).tolist()
            print("|W|", map(lambda param: float('%0.2f' % param), params))
            print()

            # Draw plots
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            mat = plt.matshow(D, fignum=0, cmap=plt.cm.gray)
            plt.colorbar(mat, fraction=0.045)
            plt.subplot(1, 2, 2)
            mat = plt.matshow(D_test, fignum=0, cmap=plt.cm.gray)
            plt.colorbar(mat, fraction=0.045)
            plt.tight_layout()

            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.plot(logger.loss_log, label="tr-loss")
            plt.grid()
            plt.legend(loc='best')
            plt.subplot(1, 2, 2)
            plt.plot(logger.train_log)
            plt.plot(logger.test_log)
            plt.grid()
            plt.legend(["tr-soft", "tr-hard", "tr-retr",
                        "te-soft", "te-hard", "te-retr"],
                       bbox_to_anchor=(1.4, 1))
            plt.ylim([0.0, 1.0])
            plt.xlim([0, p.num_epochs])
            plt.tight_layout()
            plt.show()
            plt.draw()

            loss = None
            D = None
            D_test = None

    except KeyboardInterrupt:
        pass

    dir_name = "-".join([script_filename, time.strftime("%Y%m%d%H%M%S"),
                         str(logger.soft_test_best[0])])

    logger.save(dir_name)
    p.save(dir_name)

    print("total epochs: {} ({} [s])".format(logger.epoch, logger.total_time))
    print("best test score (at # {})".format(logger.epoch_best))
    print("[test]  soft:", logger.soft_test_best)
    print("[test]  hard:", logger.hard_test_best)
    print("[test]  retr:", logger.retrieval_test_best)
    print(str(p).replace(', ', '\n'))
    print()


if __name__ == '__main__':
    random_state = None
    num_runs = 10000
    param_distributions = dict(
        learning_rate=LogUniformDistribution(low=1e-5, high=1e-4),
        loss_l2_reg=LogUniformDistribution(low=1e-7, high=1e-1),
        l2_weight_decay=LogUniformDistribution(low=1e-5, high=1e-2),
        optimizer=['RMSProp', 'Adam']  # 'RMSPeop' or 'Adam'
    )
    static_params = dict(
        num_epochs=20,
        num_batches_per_epoch=500,
        batch_size=120,
        out_dim=64,
#        learning_rate=7.10655234311e-05,
#        loss_l2_reg=2.80690151536e-06,  # L2-norm penalty for output vector
        crop_size=224,
        normalize_output=False,
#        l2_weight_decay=0.00579416451873,
#        optimizer='Adam',  # 'Adam' or 'RMSPeop'
        distance_type='euclidean',  # 'euclidean' or 'cosine'
        dataset='cars196'  # 'cars196' or 'cub200_2011' or 'products'
    )

    sampler = ParameterSampler(param_distributions, num_runs, random_state)

    for random_params in sampler:
        params = {}
        params.update(random_params)
        params.update(static_params)

        main(params)
