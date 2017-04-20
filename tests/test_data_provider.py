# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 20:30:26 2017

@author: sakurai
"""

import unittest

import numpy as np

from ..datasets.data_provider import NPairLossScheme
from ..datasets.data_provider import EpochwiseShuffledInfiniteScheme


class TestNPairLossScheme(unittest.TestCase):

    def test_pairs_of_indexes(self):
        batch_size = 20
        labels = sum([[i]*10 for i in range(10)], [])
        scheme = NPairLossScheme(labels, batch_size)
        it = scheme.get_request_iterator()
        for i in range(5):
            indexes = next(it)
            a_indexes = indexes[:batch_size // 2]
            p_indexes = indexes[batch_size // 2:]
            a_labels = np.array(labels)[a_indexes]
            p_labels = np.array(labels)[p_indexes]

            np.testing.assert_array_equal(a_labels, p_labels)
            np.testing.assert_equal(len(a_labels), len(np.unique(a_labels)))


class TestEpochwiseShuffledInfiniteScheme(unittest.TestCase):

    def check_generate_valid_indexes(self, num_examples, batch_size):
        T = 90
        scheme = EpochwiseShuffledInfiniteScheme(num_examples, batch_size)
        uniquenesses = []
        all_indexes = []
        for i in range(T):
            indexes = next(scheme)
            is_unique = len(indexes) == len(np.unique(indexes))
            uniquenesses.append(is_unique)
            all_indexes.append(indexes)

        assert np.all(uniquenesses)

        counts = np.bincount(np.concatenate(all_indexes).ravel())
        expected_counts = [batch_size * T // num_examples] * num_examples
        assert np.array_equal(counts, expected_counts)

    def test_generate_valid_indexes(self):
        self.check_generate_valid_indexes(9, 1)
        self.check_generate_valid_indexes(9, 2)
        self.check_generate_valid_indexes(9, 3)
        self.check_generate_valid_indexes(9, 8)
        self.check_generate_valid_indexes(9, 9)

        self.check_generate_valid_indexes(10, 1)
        self.check_generate_valid_indexes(10, 2)
        self.check_generate_valid_indexes(10, 3)
        self.check_generate_valid_indexes(10, 9)
        self.check_generate_valid_indexes(10, 10)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            EpochwiseShuffledInfiniteScheme(100, 101)
        with self.assertRaises(ValueError):
            EpochwiseShuffledInfiniteScheme([0, 1, 2, 0], 2)

if __name__ == '__main__':
    unittest.main()
