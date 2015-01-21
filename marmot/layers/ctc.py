from softmax import Softmax

import numpy
import theano
import theano.tensor as T

import marmot.ctc as ctc

class CTC(Softmax):

    def __init__(self, prev_layer, n):
        super(Softmax, self).__init__(prev_layer, n+1)

    def cost(self, dataset):
        return ctc.cost(
            self.activations(dataset.inputs),
            dataset.ttargets,
            dataset.target_lengths
        )

    def accuracy(self, dataset):
        return ctc.accuracy(
            self.activations(dataset.inputs),
            dataset.targets,
            dataset.target_lengths
        )