import numpy
import theano
import theano.tensor as T

from sequence_dataset import SequenceDataset

class FlatDataset(SequenceDataset):

    def __init__(
        self, 
        inputs, 
        targets, 
        minibatch_size=128,
        ):
        """Load a flat (non-sequence) dataset into Theano shared variables."""

        # Convert to numpy arrays
        inputs = numpy.array(inputs, dtype=theano.config.floatX)
        targets = numpy.array(targets, dtype=theano.config.floatX)

        # Convert to length-1 sequences
        inputs = inputs.reshape((inputs.shape[0], 1) + inputs.shape[1:])
        targets = targets.reshape((targets.shape[0], 1) + targets.shape[1:])

        # Initialize as a SequenceDataset
        super(FlatDataset, self).__init__(inputs, targets, minibatch_size)