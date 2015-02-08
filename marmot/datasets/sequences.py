import numpy
import theano
import theano.tensor as T

import marmot.ctc as ctc
import helpers

class SequenceMinibatch(object):
    def __init__(self, lengths, inputs, targets):
        self.lengths = lengths
        self.inputs = inputs
        self.targets = targets

class Sequences(object):

    def __init__(
        self, 
        inputs, 
        targets, 
        minibatch_size=128,
        input_padding_val=0,
        target_padding_val=ctc.PADDING
        ):
        """Load a dataset into Theano shared variables."""

        # Apply padding
        inputs, lengths = helpers.pad(inputs, input_padding_val)
        targets, _ = helpers.pad(targets, target_padding_val)

        # Shuffle
        inputs, \
        targets, \
        lengths = helpers.parallel_shuffle(
            (inputs, targets, lengths)
        )

        # Now sort in order of increasing input length
        inputs, \
        targets, \
        lengths = helpers.sort_by(
            lengths,
            (inputs, targets, lengths)
        )

        # Transpose data from example -> sequence to sequence -> example
        inputs = inputs.transpose(1, 0, 2)
        # Does each example have a target class (instead of an output vector)?
        classification = (len(targets.shape) == 2)
        if classification:
            targets = targets.transpose(1, 0)
        else:
            targets = targets.transpose(1, 0, 2)

        # Load into theano shared vars
        self.inputs = theano.shared(inputs, borrow=True)
        self.targets = theano.shared(targets, borrow=True)
        # this needs to be stored on the GPU as floats and casted when needed
        self.lengths = theano.shared(
            lengths.astype(theano.config.floatX), 
            borrow=True)

        self.minibatch_size = minibatch_size
        self.minibatch_count = int(numpy.ceil(float(inputs.shape[1]) / minibatch_size))

        self.ttargets = ctc.transform_targets(self.targets)

    def minibatch(self, index):
        mb_start = self.minibatch_size * index
        mb_end = T.min((self.minibatch_size * (index + 1), self.inputs.shape[1]), axis=0)

        lengths = T.cast(self.lengths[mb_start:mb_end], 'int32')
        # Explicitly specify axis to work around Theano bug
        mb_max_len = T.max(lengths, axis=0)

        inputs = self.inputs[:mb_max_len, mb_start:mb_end]
        targets = self.targets[:mb_max_len, mb_start:mb_end]
        
        return SequenceMinibatch(
            lengths,
            inputs,
            targets
        )