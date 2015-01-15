import unittest
import numpy
import theano
import theano.tensor as T

import helpers
from marmot import helpers as m_helpers

class HelpersTest(unittest.TestCase):

    def test_right_shift(self):
        shift = m_helpers.right_shift(numpy.array([[1,2,3], [4,5,6]]), 1, -1)
        correct = [
                [-1,-1,-1],
                [1, 2, 3]
            ]
        helpers.assert_theano_equal(shift, correct)

    def test_right_shift_rows(self):
        shift = m_helpers.right_shift_rows(numpy.array([[1,2,3], [4,5,6]]), 1, -1)
        correct = [
                [-1, 1, 2],
                [-1, 4, 5]
            ]
        helpers.assert_theano_equal(shift, correct)