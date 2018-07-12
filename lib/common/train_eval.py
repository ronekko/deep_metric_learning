# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:40:25 2017

@author: sakurai
"""

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import six
import time

import chainer
from chainer import cuda
from chainer import Variable
from tqdm import tqdm

from . import utils
from ..datasets import data_provider
from ..models.modified_googlenet import ModifiedGoogLeNet


def train(main_script_path, func_train_one_batch, param_dict,
          save_distance_matrix=False,):
    script_filename = os.path.splitext(os.path.basename(main_script_path))[0]

    chainer.config.train = False
    device = 0
    xp = chainer.cuda.cupy
    config_parser = six.moves.configparser.ConfigParser()
    config_parser.read('config')
    log_dir_path = os.path.expanduser(config_parser.get('logs', 'dir_path'))

    p = utils.Logger(log_dir_path, **param_dict)  # hyperparameters

    ##########################################################
    # load database
    ##########################################################
    if p.method == 'proxy_nca':
        iteration_scheme = 'clustering'
    else:
        iteration_scheme = p.method
    streams = data_provider.get_streams(p.batch_size, dataset=p.dataset,
                                        method=iteration_scheme,
                                        crop_size=p.crop_size)
    stream_train, stream_train_eval, stream_test = streams
    iter_train = stream_train.get_epoch_iterator()

    ##########################################################
    # construct the model
    ##########################################################
    if p.method == 'proxy_nca':
        dataset_class = data_provider.get_dataset_class(p.dataset)
        labels = dataset_class(
            ['train'], sources=['targets'], load_in_memory=True).data_sources
        num_classes = len(np.unique(labels))
        model = ModifiedGoogLeNet(p.out_dim, p.normalize_output, num_classes)
    else:
        model = ModifiedGoogLeNet(p.out_dim, p.normalize_output)

    if device >= 0:
        model.to_gpu()
    model.cleargrads()
    xp = model.xp
    optimizer_class = getattr(chainer.optimizers, p.optimizer)
    optimizer = optimizer_class(p.learning_rate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(p.l2_weight_decay))

    print(p)
    stop = False
    logger = utils.Logger(log_dir_path)
    logger.soft_test_best = [0]
    time_origin = time.time()
    try:
        for epoch in range(p.num_epochs):
            time_begin = time.time()
            epoch_losses = []

            for i in tqdm(range(p.num_batches_per_epoch),
                          desc='# {}'.format(epoch)):
                with chainer.using_config('train', True):
                    loss = func_train_one_batch(model, p, next(iter_train))
                    loss.backward()
                optimizer.update()
                model.cleargrads()
                epoch_losses.append(loss.data)
                del loss

            loss_average = cuda.to_cpu(xp.array(
                xp.hstack(epoch_losses).mean()))

            # average accuracy and distance matrix for training data
            D, soft, hard, retrieval = evaluate(
                model, stream_train_eval.get_epoch_iterator(), p.distance_type,
                return_distance_matrix=save_distance_matrix)

            # average accuracy and distance matrix for testing data
            D_test, soft_test, hard_test, retrieval_test = evaluate(
                model, stream_test.get_epoch_iterator(), p.distance_type,
                return_distance_matrix=save_distance_matrix)

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
            if save_distance_matrix:
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

            del D
            del D_test

    except KeyboardInterrupt:
        stop = True

    dir_name = "-".join([p.dataset, script_filename,
                         time.strftime("%Y%m%d%H%M%S"),
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

    return stop


def iterate_forward(model, epoch_iterator, normalize=False):
    xp = model.xp
    y_batches = []
    c_batches = []
    for batch in tqdm(copy.copy(epoch_iterator)):
        x_batch_data, c_batch_data = batch
        x_batch = Variable(xp.asarray(x_batch_data))
        y_batch = model(x_batch)
        if normalize:
            y_batch_data = y_batch.data / xp.linalg.norm(
                y_batch.data, axis=1, keepdims=True)
        else:
            y_batch_data = y_batch.data
        y_batches.append(y_batch_data)
        y_batch = None
        c_batches.append(c_batch_data)
    y_data = cuda.to_cpu(xp.concatenate(y_batches))
    c_data = np.concatenate(c_batches)
    return y_data, c_data


# memory friendly average accuracy for test data
def evaluate(model, epoch_iterator, distance='euclidean', normalize=False,
             batch_size=10, return_distance_matrix=False):
    if distance not in ('cosine', 'euclidean'):
        raise ValueError("distance must be 'euclidean' or 'cosine'.")

    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            y_data, c_data = iterate_forward(
                    model, epoch_iterator, normalize=normalize)

    add_epsilon = True
    xp = cuda.get_array_module(y_data)
    num_examples = len(y_data)

    D_batches = []
    softs = []
    hards = []
    retrievals = []
    yy = xp.sum(y_data ** 2.0, axis=1)

    if distance == 'cosine':
        y_data = y_data / yy[:, None]  # L2 normalization

    for start in range(0, num_examples, batch_size):
        end = start + batch_size
        if end > num_examples:
            end = num_examples
        y_batch = y_data[start:end]
        yy_batch = yy[start:end]
        c_batch = c_data[start:end]

        D_batch = yy + yy_batch[:, None] - 2.0 * xp.dot(y_batch, y_data.T)
        xp.maximum(D_batch, 0, out=D_batch)
        if add_epsilon:
            D_batch += 1e-40
        # ensure the diagonal components are zero
        xp.fill_diagonal(D_batch[:, start:end], 0)

        soft, hard, retr = compute_soft_hard_retrieval(
            D_batch, c_data, c_batch)

        softs.append(len(y_batch) * soft)
        hards.append(len(y_batch) * hard)
        retrievals.append(len(y_batch) * retr)
        if return_distance_matrix:
            D_batches.append(D_batch)

    avg_softs = xp.sum(softs, axis=0) / num_examples
    avg_hards = xp.sum(hards, axis=0) / num_examples
    avg_retrievals = xp.sum(retrievals, axis=0) / num_examples

    if return_distance_matrix:
        D = cuda.to_cpu(xp.vstack(D_batches))
    else:
        D = None
    return D, avg_softs, avg_hards, avg_retrievals


def compute_soft_hard_retrieval(distance_matrix, labels, label_batch=None):
    softs = []
    hards = []
    retrievals = []

    if label_batch is None:
        label_batch = labels
    distance_matrix = cuda.to_cpu(distance_matrix)
    labels = cuda.to_cpu(labels)
    label_batch = cuda.to_cpu(label_batch)

    K = 11  # "K" for top-K
    for d_i, label_i in zip(distance_matrix, label_batch):
        top_k_indexes = np.argpartition(d_i, K)[:K]
        sorted_top_k_indexes = top_k_indexes[np.argsort(d_i[top_k_indexes])]
        ranked_labels = labels[sorted_top_k_indexes]
        # 0th entry is excluded since it is always 0
        ranked_hits = ranked_labels[1:] == label_i

        # soft top-k, k = 1, 2, 5, 10
        soft = [np.any(ranked_hits[:k]) for k in [1, 2, 5, 10]]
        softs.append(soft)
        # hard top-k, k = 2, 3, 4
        hard = [np.all(ranked_hits[:k]) for k in [2, 3, 4]]
        hards.append(hard)
        # retrieval top-k, k = 2, 3, 4
        retrieval = [np.mean(ranked_hits[:k]) for k in [2, 3, 4]]
        retrievals.append(retrieval)

    average_soft = np.array(softs).mean(axis=0)
    average_hard = np.array(hards).mean(axis=0)
    average_retrieval = np.array(retrievals).mean(axis=0)
    return average_soft, average_hard, average_retrieval
