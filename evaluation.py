# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:47:00 2017

@author: sakurai
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import f1_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder


def ap_cluster_k(x, K, preference_init=-1.0, max_iter=30,
                 c=None, iter_finetune=10):
    '''
    Clustering of x by affinity propagation which the number of cluster is K.

    args:
        x (ndarray):
            Data matrix.
        K (int):
            Target number of clusters.
        max_iter (int):
            Number of trials for bisection search.
        c (ndarray, optional):
            Class labels of x. If this parameter is specified, the function
            try to find the better solution by random search.
        iter_finetune (int):
            Number of steps for the random search.
    '''

    # first, search rough lower bound of the preference
    assert preference_init < 0, "preference_init must be negative."
    p = float(preference_init)  # preference parameter
    p_upper = 0
    for i in range(5):
        ap = AffinityPropagation(preference=p).fit(y)
        k_current = len(ap.cluster_centers_indices_)
        if k_current > K:
            p_upper = p
            k_upper = k_current
            p *= 10
        else:
            p_lower = p
            k_lower = k_current
            break
    else:
        raise RuntimeError("Can't find initial lower bound for preference."
                           " Try another value of p_initial.")

    # search the preference by bisection method
    for i in range(max_iter):
        p = (p_lower + p_upper) / 2
        ap = AffinityPropagation(preference=p).fit(y)
        k_current = len(ap.cluster_centers_indices_)
        print('K = {}, k_current = {}, p = {}'.format(K, k_current, p))
        print('{}:{}, {}:{}, {}:{}'.format(k_lower, p_lower, k_current, p,
                                           k_upper, p_upper))

        # if the current k goes out of bounds then retry with perturbed p
        while k_current < k_lower or k_current > k_upper:
            print("retry")
            p += np.random.uniform(p_lower, p_upper) / 10
            ap = AffinityPropagation(preference=p).fit(y)
            k_current = len(ap.cluster_centers_indices_)
            print('K = {}, k_current = {}, p = {}'.format(K, k_current, p))
            print('{}:{}, {}:{}, {}:{}'.format(k_lower, p_lower, k_current, p,
                                               k_upper, p_upper))

        if k_current < K:
            p_lower = p
            k_lower = k_current
        elif k_current > K:
            p_upper = p
            k_upper = k_current
        else:
            break
    else:
        raise RuntimeError("Can't find a preference to form K clusters."
                           " Try another value of p_initial.")

    if c is None:
        return ap

    # Search further better preference in terms of NMI score by random search
    p_best = p
    score_best = normalized_mutual_info_score(c, ap.predict(y))
    print('initial score:', score_best)
    print()
    for i in range(iter_finetune):
        p = np.random.normal(p_best, (p_upper - p_lower) / 2)
        if p < p_lower or p > p_upper:  # where p is rejected
            print('reject')
            continue
        ap = AffinityPropagation(preference=p).fit(y)
        k_current = len(ap.cluster_centers_indices_)
        if k_current < K and p > p_lower:
            p_lower = p
        elif k_current > K and p < p_upper:
            p_upper = p
        else:  # wgere k_current is K
            score = normalized_mutual_info_score(c, ap.predict(y))
            if score > score_best:
                print("update p {} -> {}".format(p_best, p))
                p_best = p
                score_best = score
        print('p: {}, {}, {}'.format(p_lower, p, p_upper))
        print('score: {}'.format(score_best))
        print()
    return AffinityPropagation(preference=p_best).fit(y)


if __name__ == '__main__':
    y_train = np.load('y_train.npy')
    c_train = np.load('c_train.npy').ravel()
    y_test = np.load('y_test.npy')
    c_test = np.load('c_test.npy').ravel()

    c_train = LabelEncoder().fit_transform(c_train)
    c_test = LabelEncoder().fit_transform(c_test)

    K = 40
#    K = len(np.unique(c_train))
    y = y_train[c_train.ravel() < K]
    c = c_train[c_train < K]
#    y = y_test[c_test.ravel() < K]
#    c = c_test[c_test < K]

    ap = ap_cluster_k(y, K, preference_init=-1.0, c=c, iter_finetune=30)
    c_pred = ap.predict(y)

    print(normalized_mutual_info_score(c, c_pred))
    plt.plot(np.vstack((c_pred, c)).T)
    plt.show()
#    print f1_score(c, c_pred)
