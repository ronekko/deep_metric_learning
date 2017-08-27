# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:39:32 2017

@author: sakurai
"""

from collections import defaultdict
from glob import glob
import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import six
from sklearn.preprocessing import LabelEncoder


def cast_if_number(string):
    try:
        return float(string)
    except ValueError:
        return string


def read_params(target_prefix, begin=None, end=None):
    if os.path.exists('config'):
        config_parser = six.moves.configparser.ConfigParser()
        config_parser.read('config')
        log_dir_path = os.path.expanduser(
            config_parser.get('logs', 'dir_path'))
    else:
        log_dir_path = '.'

    if begin:
        begin = time.strptime(begin, "%Y%m%d%H%M%S")
    if end:
        end = time.strptime(end, "%Y%m%d%H%M%S")

    params = []
    for dir_path in glob(os.path.join(log_dir_path, target_prefix) + '*'):
        dir_name = dir_path.split('\\')[-1]
        score = dir_name.split('-')[-1]
        timestamp = dir_name.split('-')[-2]
        score = float(score)
        timestamp = time.strptime(timestamp, "%Y%m%d%H%M%S")
        if begin and timestamp < begin:
            continue
        if end and timestamp >= end:
            continue

        param_dict = dict(score=score)
        with open(os.path.join(dir_path, 'log.txt')) as f:
            for line in f.readlines():
                key, value = line.split(': ')
                param_dict[key] = cast_if_number(value.strip())
        params.append(param_dict)

    paramwise = defaultdict(list)
    for p in params:
        for k, v in p.items():
            paramwise[k].append(v)
    return paramwise


def read_learning_curves(target_prefix, begin=None, end=None):
    if os.path.exists('config'):
        config_parser = six.moves.configparser.ConfigParser()
        config_parser.read('config')
        log_dir_path = os.path.expanduser(
            config_parser.get('logs', 'dir_path'))
    else:
        log_dir_path = '.'

    if begin:
        begin = time.strptime(begin, "%Y%m%d%H%M%S")
    if end:
        end = time.strptime(end, "%Y%m%d%H%M%S")

    curves = []
    for dir_path in glob(os.path.join(log_dir_path, target_prefix) + '*'):
        dir_name = dir_path.split('\\')[-1]
        timestamp = dir_name.split('-')[-2]
        timestamp = time.strptime(timestamp, "%Y%m%d%H%M%S")
        if begin and timestamp < begin:
            continue
        if end and timestamp >= end:
            continue

        test_log = np.load(os.path.join(dir_path, 'test_log.npy'))
        curves.append(test_log.T[0])

    return curves


if __name__ == '__main__':
    target_prefix = 'cub200_2011-main_n_pair_mc'

#    begin = '20170822121156'
    begin = '20170823203509'
    begin = '20170825130000'
    begin = '20170826143546'
    end = None

    params = read_params(target_prefix, begin, end)
    curves = read_learning_curves(target_prefix, begin, end)

#    # filtering
#    for i in list(range(len(curves)))[::-1]:
#        if curves[i][:-2].max() > curves[i][-2:].max():
#            curves.pop(i)
#            for param in params.values():
#                param.pop(i)

    # Show learning curves
    for curve in curves:
        plt.plot(curve, linewidth=1)
    plt.grid()
    plt.show()

    # Show scatter plots of parameters
    scores = params['score']

    param_name = 'learning_rate'
    learning_rates = params[param_name]
    plt.plot(learning_rates, scores, '.')
    plt.title(param_name)
    plt.xlabel(param_name)
    plt.ylabel('score')
    plt.xscale('log')
    plt.grid()
    plt.show()

    param_name = 'loss_l2_reg'
    loss_l2_reg = params[param_name]
    plt.plot(loss_l2_reg, scores, '.')
    plt.title(param_name)
    plt.xlabel(param_name)
    plt.ylabel('score')
    plt.xscale('log')
    plt.grid()
    plt.show()

    param_name = 'optimizer'
    optimizers = params[param_name]
    le = LabelEncoder().fit(optimizers)
    plt.xticks(range(len(le.classes_)), le.classes_)
    plt.plot(le.transform(optimizers), scores, '.')
    plt.title(param_name)
    plt.xlabel(param_name)
    plt.ylabel('score')
    plt.grid()
    plt.show()

    param_name = 'l2_weight_decay'
    l2_weight_decays = params[param_name]
    plt.plot(l2_weight_decays, scores, '.')
    plt.title(param_name)
    plt.xlabel(param_name)
    plt.ylabel('score')
    plt.xscale('log')
    plt.grid()
    plt.show()

    param_name = 'out_dim'
    out_dim = params[param_name]
    plt.xticks(list(set(out_dim)))
    plt.plot(out_dim, scores, '.')
    plt.title(param_name)
    plt.xlabel(param_name)
    plt.ylabel('score')
#    plt.xscale('log')
    plt.grid()
    plt.show()

    # scatter plot on learning rate and loss-specific hyperparameter
    plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('learning_rate')
    plt.ylabel('loss_l2_reg')
    plt.grid()
    plt.scatter(learning_rates, loss_l2_reg, c=scores, cmap=plt.cm.rainbow)
    plt.colorbar()
    plt.show()

    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.log10(learning_rates), np.log10(loss_l2_reg), scores,
               c=scores, cmap=plt.cm.rainbow)
    plt.xlabel('learning_rate in common log')
    plt.ylabel('loss_l2_reg in common log')
    plt.show()
