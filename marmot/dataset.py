import numpy
import theano
import theano.tensor as T

class Dataset(object):

    def __init__(self, inputs, targets):
        """Load a dataset into Theano shared variables."""

        # Convert to numpy arrays
        inputs = numpy.array(inputs, dtype=theano.config.floatX)
        targets = numpy.array(targets, dtype=theano.config.floatX)

        # If the dataset is flat, turn it into length-1 sequences
        if len(inputs.shape) == 2:
            inputs = inputs.reshape((inputs.shape[0], 1) + inputs.shape[1:])
            targets = targets.reshape((targets.shape[0], 1) + targets.shape[1:])

        # Does each example have a target class (instead of a vector)?
        self.classification = (len(targets.shape) == 2)

        self.example_count = inputs.shape[0]
        self.input_sequence_length = inputs.shape[1]
        self.input_length = inputs.shape[2]
        self.target_sequence_length = targets.shape[1]
        if not self.classification:
            self.target_length = targets.shape[2]

        # Transpose data from example -> sequence to sequence -> example
        inputs = inputs.transpose(1, 0, 2)
        if self.classification:
            targets = targets.transpose(1, 0)
        else:
            targets = targets.transpose(1, 0, 2)

        # Convert to theano shared vars
        self.inputs = theano.shared(inputs, borrow=True)
        _uncast_targets = theano.shared(targets, borrow=True)

        # If the targets represent classes, then they need to be integers
        # because they'll be used as indices into an output vector.
        # They need to be stored as float32 on the GPU, though, so we
        # cast them to ints here.
        if self.classification:
            self.targets = T.cast(_uncast_targets, 'int32')
        else:
            self.targets = _uncast_targets

        # Define a Theano shuffle function: shuffle inputs and targets in unison
        # along the example dimension.
        rng = T.shared_randomstreams.RandomStreams()
        permutation = rng.permutation((), self.inputs.shape[1])
        shuffled_inputs = self.inputs[:, permutation]
        shuffled_targets = _uncast_targets[:, permutation]

        self.shuffle = theano.function([], [], updates=(
            (self.inputs, shuffled_inputs), 
            (_uncast_targets, shuffled_targets)
            ))