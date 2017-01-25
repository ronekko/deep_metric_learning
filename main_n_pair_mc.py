# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 20:49:04 2017

@author: sakurai
"""

import os
import time
import copy
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

colorama.init()

class Model(chainer.Chain):  # same as classifier
    def __init__(self, out_dim):
        super(Model, self).__init__(
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

    def __call__(self, x, test=True):
        h = self.conv1(x)
        h = self.bn_conv1(h, test=test)
        h = F.max_pooling_2d(h, 2)
        h = F.relu(h)

        h = self.conv21(h)
        h = self.bn_conv21(h, test=test)
        h = F.relu(h)
        h = self.conv22(h)
        h = self.bn_conv22(h, test=test)
        h = F.max_pooling_2d(h, 2)
        h = F.relu(h)

        h = self.conv31(h)
        h = self.bn_conv31(h, test=test)
        h = F.relu(h)
        h = self.conv32(h)
        h = self.bn_conv32(h, test=test)
        h = F.max_pooling_2d(h, 2)
        h = F.relu(h)

        h = self.conv41(h)
        h = self.bn_conv41(h, test=test)
        h = F.relu(h)
        h = self.conv42(h)
        h = self.bn_conv42(h, test=test)
        h = F.max_pooling_2d(h, 2)
        h = F.relu(h)

        h = self.conv5(h)
        h = self.bn_conv5(h, test=test)
        h = F.relu(h)

        h = self.conv6(h)
        h = self.bn_conv6(h, test=test)
        h = F.relu(h)

        h = F.average_pooling_2d(h, h.shape[2:])
        h = self.linear1(h)
        h = self.bn_linear1(h, test=test)
#        h = F.dropout(h, ratio=0.5, train=not test)
        h = F.relu(h)
        h = self.linear2(h)
        return h


def worker_load_data(queue, stream):
    infinite_iterator = stream.get_epoch_iterator()
    while True:
        x_data, c_data = next(infinite_iterator)
        queue.put((x_data, c_data))


if __name__ == '__main__':
    script_filename = os.path.splitext(os.path.basename(__file__))[0]
    device = 0
    xp = chainer.cuda.cupy
    learning_rate = 0.00001  # 0.00001 is good
    batch_size = 50
    out_dim = 50
    loss_l2_reg = 0.001
    crop_size = 227
    num_epochs = 5000
    num_batches_per_epoch = 500

    ##########################################################
    # load database
    ##########################################################
    iters = chainer_datasets.get_iterators(batch_size)
    iter_train, iter_train_eval, iter_test = iters
#    num_batches_per_epoch = iter_train._order_sampler.num_batches

    ##########################################################
    # construct the model
    ##########################################################
    model = Model(out_dim).to_gpu()
    model = model.to_gpu()
    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)

    loss_log = []
    train_log = []
    test_log = []
    soft_test_best = [0]
    time_origin = time.time()
    try:
        for epoch in range(num_epochs):
            time_begin = time.time()
            epoch_losses = []

            for i in tqdm(range(num_batches_per_epoch)):
                # the first half　of a batch are the anchors and the latters
                # are the positive examples corresponding to each anchor
                batch = next(iter_train)
                x_data, c_data = concat_examples(batch, device)
                y = model(x_data, test=False)
                y_a, y_p = F.split_axis(y, 2, axis=0)

                loss = n_pair_mc_loss(y_a, y_p, loss_l2_reg)
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
            # print norms of the weights
            print "|W|", [np.linalg.norm(p.data.get()) for p in model.params()]
            print "lr:{}, margin:{}".format(learning_rate, loss_l2_reg)
            print
            loss_log.append(loss_average)
            train_log.append([soft[0], hard[0], retrieval[0]])
            test_log.append([soft_test[0], hard_test[0], retrieval_test[0]])

            # retain the model if it scored the best test acc. ever
            if soft_test[0] > soft_test_best[0]:
                model_best = copy.deepcopy(model)
                optimizer_best = copy.deepcopy(optimizer)
                epoch_best = epoch
                D_best = D
                D_test_best = D_test
                soft_best = soft
                soft_test_best = soft_test
                hard_best = hard
                hard_test_best = hard_test
                retrieval_best = retrieval
                retrieval_test_best = retrieval_test

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
            plt.plot(loss_log, label="tr-loss")
            plt.grid()
            plt.legend(loc='best')
            plt.subplot(1, 2, 2)
            plt.plot(train_log)
            plt.plot(test_log)
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
                         str(soft_test_best[0])])
    os.mkdir(dir_name)
    model_path = os.path.join(dir_name, "model.npz")
    serializers.save_npz(model_path, model_best)
    optimizer_path = os.path.join(dir_name, "optimizer.npz")
    serializers.save_npz(optimizer_path, optimizer_best)
    np.save(os.path.join(dir_name, "loss_log.npy"), loss_log)
    np.save(os.path.join(dir_name, "train_log.npy"), train_log)
    np.save(os.path.join(dir_name, "test_log.npy"), test_log)
    np.save(os.path.join(dir_name, "D.npy"), D_best)
    np.save(os.path.join(dir_name, "D_test.npy"), D_test_best)
    np.save(os.path.join(dir_name, "soft.npy"), soft_best)
    np.save(os.path.join(dir_name, "soft_test.npy"), soft_test_best)
    np.save(os.path.join(dir_name, "hard.npy"), hard_best)
    np.save(os.path.join(dir_name, "hard_test.npy"), hard_test_best)
    np.save(os.path.join(dir_name, "retrieval.npy"), retrieval_best)
    np.save(os.path.join(dir_name, "retrieval_test.npy"), retrieval_test_best)
    with open(os.path.join(dir_name, "log.txt"), "w") as f:
        text = "\n".join(["total epochs: {}".format(epoch),
                          "best epoch: {}".format(epoch_best),
                          "total time: {} [s]".format(total_time),
                          "lr: {}".format(learning_rate),
                          "batch_size: {}".format(batch_size),
                          "loss_l2_reg: {}".format(loss_l2_reg),
                          "out_dim: {}".format(out_dim),
                          "crop_size: {}".format(crop_size)])
        f.write(text)
