# import numpy
# import theano
# import theano.tensor as T

# from sequences import Sequences

# class Simple(Sequences):
#     """Dataset class for representing 'simple' datasets 
#        (labelled fixed vectors; e.g. MNIST)."""

#     def __init__(
#         self, 
#         inputs, 
#         targets, 
#         minibatch_size=128,
#         ):
#         """Load a flat (non-sequence) dataset into Theano shared variables."""

#         # Convert to numpy arrays
#         inputs = numpy.array(inputs, dtype=theano.config.floatX)
#         targets = numpy.array(targets, dtype=theano.config.floatX)

#         # Convert to length-1 sequences
#         inputs = inputs.reshape((inputs.shape[0], 1) + inputs.shape[1:])
#         targets = targets.reshape((targets.shape[0], 1) + targets.shape[1:])

#         # Initialize as a SequenceDataset
#         super(Simple, self).__init__(inputs, targets, minibatch_size)

import numpy
import theano
import theano.tensor as T

import helpers

class SimpleMinibatch(object):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

class Simple(object):
    """Dataset class for representing 'simple' datasets 
       (labelled fixed vectors; e.g. MNIST)."""

    def __init__(
        self,
        inputs,
        targets,
        minibatch_size=128,
        shuffle=True
        ):
        """Load a flat (non-sequence) dataset into Theano shared variables."""

        # Convert to numpy arrays
        inputs = numpy.array(inputs, dtype=theano.config.floatX)
        targets = numpy.array(targets, dtype=theano.config.floatX)

        # Shuffle
        if shuffle:
            inputs, targets = helpers.parallel_shuffle((inputs, targets))

        # Load into Theano shared vars
        self.inputs = theano.shared(inputs, borrow=True)
        self.targets = theano.shared(targets, borrow=True)

        self.minibatch_size = minibatch_size
        self.minibatch_count = int(numpy.ceil(float(inputs.shape[0]) / minibatch_size))

    def minibatch(self, index):
        mb_start = self.minibatch_size * index
        mb_end = T.min((self.minibatch_size * (index + 1), self.inputs.shape[0]), axis=0)

        inputs = self.inputs[mb_start:mb_end]
        targets = self.targets[mb_start:mb_end]
        
        return SimpleMinibatch(inputs,targets)