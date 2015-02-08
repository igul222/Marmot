import unittest
import numpy
import theano

from marmot.datasets import Sequence

class SequenceTest(unittest.TestCase):

    # def test_create_flat(self):
    #     inputs = [[1,3,5], [2,4,6]]
    #     targets = [0,1]
    #     dataset = Dataset(inputs, targets)

    #     inputs = theano.function([],[dataset.inputs])()[0]
    #     targets = theano.function([],[dataset.targets])()[0]

    #     self.assertTrue(numpy.array_equal(
    #         inputs, 
    #         [[[1,3,5], [2,4,6]]]
    #     ))
    #     self.assertTrue(numpy.array_equal(
    #         targets,
    #         [[0,1]]
    #     ))

    def test_create_sequence(self):
        inputs = [[[1],[0],[0]], [[0], [0], [0]]]
        targets = [[1,0,0], [0,0,0]]
        dataset = SequenceDataset(inputs, targets)

        inputs = theano.function([],[dataset.inputs])()[0]
        targets = theano.function([],[dataset.targets])()[0]

        self.assertTrue(numpy.array_equal(
            inputs,
            [[[1],[0]],[[0],[0]],[[0],[0]]]
        ))
        self.assertTrue(numpy.array_equal(
            targets,
            [[1,0], [0,0], [0,0]]
        ))

    def test_create_float_targets(self):
        pass

    def test_shuffle(self):
        inputs = [[1,2,3],[4,5,6],[7,8,9]]
        targets = [2,5,8]
        dataset = Dataset(inputs, targets)
        dataset.shuffle()

        inputs = theano.function([],[dataset.inputs])()[0]
        targets = theano.function([],[dataset.targets])()[0]

        self.assertEqual(inputs[0][0][1], targets[0][0])
        self.assertEqual(inputs[0][1][1], targets[0][1])
        self.assertEqual(inputs[0][2][1], targets[0][2])

    def test_variable_lengths(self):
        inputs = [[[1],[0],[0]], 
                  [[0],[1]]]
        targets = [[1,0,0], [0,1]]
        dataset = Dataset(inputs, targets)