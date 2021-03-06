import unittest
import numpy
import theano
import theano.tensor as T

import helpers
from marmot import ctc

class CTCTest(unittest.TestCase):

    def test_log_add(self):
        helpers.assert_theano_almost_equal(
            ctc._log_add(numpy.log(2), numpy.log(3)),
            numpy.log(2 + 3)
        )

    def test_log_add_3(self):
        helpers.assert_theano_almost_equal(
            ctc._log_add_3(numpy.log(2), numpy.log(3), numpy.log(4)),
            numpy.log(2 + 3 + 4)
        )

    def test_initial_probabilities(self):
        helpers.assert_theano_equal(
            ctc._initial_probabilities(2, 3),
            [[0, ctc._LOG_ZERO, ctc._LOG_ZERO], 
             [0, ctc._LOG_ZERO, ctc._LOG_ZERO]]
        )

    def test_skip_allowed(self):
        l0 = ctc._LOG_ZERO
        l1 = ctc._LOG_ONE

        helpers.assert_theano_equal(
            ctc._skip_allowed(
                numpy.array([[0, 1, 0, 2, 0], 
                             [0, 1, 0, 1, 0]])
            ),
            [[l0, l1, l0, l1, l0],
             [l0, l0, l0, l1, l0]]
        )

    def test_forward_vars(self):
        l0 = ctc._LOG_ZERO
        l1 = ctc._LOG_ONE

        activations = numpy.array([
            [[0.01, 0.97, 0.01, 0.01]],
            [[0.01, 0.01, 0.97, 0.01]],
            [[0.97, 0.01, 0.01, 0.01]],
            [[0.01, 0.01, 0.01, 0.97]],
            ], dtype=theano.config.floatX)

        targets = numpy.array([[0,1,0,2,0,3,0]], dtype=theano.config.floatX)

        helpers.assert_theano_almost_equal(
            T.exp(ctc._forward_vars(activations, targets)),
            [[[ 0.01,        0.97000003,  0.,          0.,          0.,          0.,          0.        ]],
             [[ 0.0001,      0.0098,      0.0097,      0.94090003,  0.,          0.,          0.        ]],
             [[ 0.000097,    0.000099,    0.01891499,  0.009604,    0.91267306,  0.009409,    0.        ]],
             [[ 0.00000097,  0.00000196,  0.00019014,  0.00028618,  0.00922277,  0.90373552,  0.00009409]]]
        )

    def test_cost(self):
        activations = numpy.array([
            [[0.01, 0.97, 0.01, 0.01]],
            [[0.01, 0.01, 0.97, 0.01]],
            [[0.97, 0.01, 0.01, 0.01]],
            [[0.01, 0.01, 0.01, 0.97]],
            ], dtype=theano.config.floatX)

        ttargets = numpy.array([[0,1,0,2,0,3,0]], dtype=theano.config.floatX)

        target_lengths = numpy.array([3], dtype=theano.config.floatX)

        helpers.assert_theano_almost_equal(
            ctc.cost(activations, ttargets, target_lengths),
            0.101114414632
        )

    def test_cost_padded_targets(self):
        activations = numpy.array([
            [[0.01, 0.98, 0.01], [0.01, 0.01, 0.98]],
            [[0.98, 0.01, 0.01], [0.01, 0.98, 0.01]]
        ], dtype=theano.config.floatX)

        P = ctc.PADDING
        ttargets = numpy.array([
            [0,1,0,P,0],
            [0,2,0,1,0]
        ], dtype=theano.config.floatX)

        target_lengths = numpy.array([1,2], dtype=theano.config.floatX)

        helpers.assert_theano_almost_equal(
            ctc.cost(activations, ttargets, target_lengths),
            0.0352776572108
        )

    def test_best_path_decode(self):
        activations = numpy.array([
            [[0,1,0], [0,1,0]],
            [[0,1,0], [0,0,1]],
            [[1,0,0], [1,0,0]],
            [[0,0,1], [0,0,1]],
            [[1,0,0], [0,1,0]],
            [[1,0,0], [1,0,0]]
        ], dtype=theano.config.floatX)
        
        P = ctc.PADDING
        helpers.assert_theano_equal(
            ctc._best_path_decode(activations),
            [[1, 1],
             [2, 2],
             [P, 2],
             [P, 1]]
        )

    def test_accuracy(self):
        P = ctc.PADDING

        activations = numpy.array([
            [[0,1,0], [0,1,0]],
            [[0,1,0], [0,0,1]],
            [[1,0,0], [1,0,0]],
            [[0,0,1], [0,0,1]],
            [[1,0,0], [1,0,0]],
            [[1,0,0], [1,0,0]]
        ], dtype=theano.config.floatX)

        targets = numpy.array([
            [1,1],
            [2,2],
            [P,2],
            [P,1]
        ], dtype=theano.config.floatX)

        target_lengths = numpy.array([2, 4], dtype=theano.config.floatX)

        helpers.assert_theano_equal(
            ctc.accuracy(activations, targets, target_lengths),
            0.875
        )

    def test_transform_targets(self):

        targets = numpy.array([
            [1,1],
            [2,2],
            [3,2],
            [4,1]
        ], dtype=theano.config.floatX)

        helpers.assert_theano_equal(
            ctc.transform_targets(targets),
            [[ 0., 0.],
             [ 1., 1.],
             [ 0., 0.],
             [ 2., 2.],
             [ 0., 0.],
             [ 3., 2.],
             [ 0., 0.],
             [ 4., 1.],
             [ 0., 0.]]
        )