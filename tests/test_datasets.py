# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 20:30:26 2017

@author: sakurai
"""

import unittest

import numpy as np

from ..datasets.data_provider import NPairLossScheme


class TestNPairLossScheme(unittest.TestCase):

    def test_pairs_of_indexes(self):
        batch_size = 20
        labels = sum([[i]*10 for i in range(10)], [])
        scheme = NPairLossScheme(labels, batch_size)
        it = scheme.get_request_iterator()
        for i in range(5):
            indexes = next(it)
            a_indexes = indexes[:batch_size / 2]
            p_indexes = indexes[batch_size / 2:]
            a_labels = np.array(labels)[a_indexes]
            p_labels = np.array(labels)[p_indexes]

            np.testing.assert_array_equal(a_labels, p_labels)
            np.testing.assert_equal(len(a_labels), len(np.unique(a_labels)))


if __name__ == '__main__':
    unittest.main()
