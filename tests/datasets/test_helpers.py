import unittest
import numpy
import theano
import tests.helpers

import marmot.datasets.helpers as helpers

class HelpersTest(unittest.TestCase):

    def test_pad(self):
        arrays = [
            [1],
            [1,2,3],
            [1,2]
        ]

        padded, lengths = helpers.pad(arrays, -1)

        numpy.testing.assert_array_equal(
            padded,
            [[1,-1,-1],
             [1, 2, 3],
             [1, 2,-1]]
        )

        numpy.testing.assert_array_equal(
            lengths,
            [1, 3, 2]
        )

    def test_parallel_shuffle(self):
        n = numpy.array([1,2,3,4,5])
        n_plus_one = n + 1

        n, n_plus_one = helpers.parallel_shuffle((n, n_plus_one))

        numpy.testing.assert_array_equal(
            n + 1,
            n_plus_one
        )