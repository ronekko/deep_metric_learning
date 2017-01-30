# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 20:49:04 2017

@author: sakurai
"""

import os
import time
import copy
import ConfigParser
from multiprocessing import Process, Queue
import numpy as np
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer.dataset.convert import concat_examples
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import colorama

from n_pair_mc_loss import n_pair_mc_loss
import common
from datasets import get_cars196_streams
import chainer_datasets
from models import ModifiedGoogLeNet

colorama.init()


def worker_load_data(queue, stream):
    infinite_iterator = stream.get_epoch_iterator()
    while True:
        x_data, c_data = next(infinite_iterator)
        queue.put((x_data, c_data))


if __name__ == '__main__':
    script_filename = os.path.splitext(os.path.basename(__file__))[0]
    device = 0
    xp = chainer.cuda.cupy
    config_parser = ConfigParser.SafeConfigParser()
    config_parser.read('config')
    log_dir_path = os.path.expanduser(config_parser.get('logs', 'dir_path'))

    p = common.Logger(log_dir_path)  # hyperparameters
    p.learning_rate = 0.0001  # 0.0001 is good
    p.batch_size = 60
    p.out_dim = 64
    p.loss_l2_reg = 0.001
    p.crop_size = 224
    p.num_epochs = 5000
    p.num_batches_per_epoch = 500

    ##########################################################
    # load database
    ##########################################################
    iters = chainer_datasets.get_iterators(p.batch_size)
    iter_train, iter_train_eval, iter_test = iters

    ##########################################################
    # construct the model
    ##########################################################
    model = ModifiedGoogLeNet(p.out_dim).to_gpu()
    model = model.to_gpu()
    optimizer = optimizers.Adam(p.learning_rate)
    optimizer.setup(model)

    logger = common.Logger(log_dir_path)
    logger.soft_test_best = [0]
    time_origin = time.time()
    try:
        for epoch in range(p.num_epochs):
            time_begin = time.time()
            epoch_losses = []

            for i in tqdm(range(p.num_batches_per_epoch)):
                # the first half　of a batch are the anchors and the latters
                # are the positive examples corresponding to each anchor
                batch = next(iter_train)
                x_data, c_data = concat_examples(batch, device)
                y = model(x_data, train=True)
                y_a, y_p = F.split_axis(y, 2, axis=0)

                loss = n_pair_mc_loss(y_a, y_p, p.loss_l2_reg)
                optimizer.zero_grads()
                loss.backward()
                optimizer.update()

                loss_data = loss.data.get()
                epoch_losses.append(loss_data)
                y = y_a = y_p = loss = None

            loss_average = np.mean(epoch_losses)

            # average accuracy and distance matrix for training data
            D, soft, hard, retrieval = common.evaluate(model, iter_train_eval)

            # average accuracy and distance matrix for testing data
            result_test = common.evaluate(model, iter_test)
            D_test, soft_test, hard_test, retrieval_test = result_test

            time_end = time.time()
            epoch_time = time_end - time_begin
            total_time = time_end - time_origin

            print "#", epoch
            print "time: {} ({})".format(epoch_time, total_time)
            print "[train] loss:", loss_average
            print "[train] soft:", soft
            print "[train] hard:", hard
            print "[train] retr:", retrieval
            print "[test]  soft:", soft_test
            print "[test]  hard:", hard_test
            print "[test]  retr:", retrieval_test
            print "lr:{}, l2_loss_reg:{}, bs:{}, out_dim:{}".format(
                p.learning_rate, p.loss_l2_reg, p.batch_size, p.out_dim)
            # print norms of the weights
            print "|W|", [np.linalg.norm(w.data.get()) for w in model.params()]
            print
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
            plt.tight_layout()
            plt.show()
            plt.draw()

            loss = None
            accuracy = None
            accuracy_test = None
            D = None
            D_test = None

    except KeyboardInterrupt:
        pass

    dir_name = "-".join([script_filename, time.strftime("%Y%m%d%H%H%S"),
                         str(logger.soft_test_best[0])])

    logger.save(dir_name)
    p.save(dir_name)
