import unittest
import numpy
import theano
import theano.tensor as T

import helpers
from marmot import levenshtein

class LevenshteinTest(unittest.TestCase):

    def test_simple(self):
        P = -1 # padding
        a = numpy.array([[1,2,3]], dtype=theano.config.floatX).T # BAT
        b = numpy.array([[4,2,3]], dtype=theano.config.floatX).T # CAT

        helpers.assert_theano_equal(
            levenshtein.distances(a,b),
            [1]
        )

    # def test_distance(self):
    #     P = -1 # padding
    #     a = numpy.array([
    #         [1,2,3,4,5,6,7,8], # islander
    #         # [1,2,3,4,P,P,P,P], # mart
    #         # [1,2,3,3,4,5,P,P], # kitten
    #         # [1,2,3,4,2,3,1,5]  # intentio [sic]
    #     ], dtype=theano.config.floatX).T
    #     b = numpy.array([
    #         [2,3,4,5,6,7,8,P,P], # slander
    #         # [5,2,3,1,2,P,P,P,P], # karma
    #         # [6,2,3,3,2,5,7,P,P], # sitting
    #         # [4,6,4,7,8,3,1,5,2]  # execution
    #     ], dtype=theano.config.floatX).T

    #     distances = [1,5,5,8]

    #     helpers.assert_theano_equal(
    #         levenshtein.distance(a,b),
    #         distances
    #     )