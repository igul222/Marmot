import numpy
import theano
import theano.tensor as T

import marmot.ctc as ctc

class SequenceMinibatch(object):
    def __init__(self, inputs, targets, input_lengths, target_lengths, ttargets):
        self.inputs = inputs
        self.targets = targets
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths
        self.ttargets = ttargets

class SequenceDataset(object):

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
        def pad(arr, val):
            lengths = [len(e) for e in arr]
            max_len = max(lengths)
            padded = [numpy.concatenate((e, numpy.empty((max_len - len(e),)+e.shape[1:])*val)) for e in arr]
            return (padded, lengths)

        (inputs, input_lengths) = pad(inputs, input_padding_val)
        (targets, target_lengths) = pad(targets, target_padding_val)

        # Convert to numpy arrays
        inputs = numpy.array(inputs, dtype=theano.config.floatX)
        targets = numpy.array(targets, dtype=theano.config.floatX)
        input_lengths = numpy.array(input_lengths)
        target_lengths = numpy.array(target_lengths)

        # Shuffle dataset
        # shuffle = numpy.arange(len(inputs))
        # numpy.random.shuffle(shuffle)
        # inputs = inputs[shuffle]
        # targets = targets[shuffle]
        # input_lengths = input_lengths[shuffle]
        # target_lengths = target_lengths[shuffle]

        # Now sort in order of increasing input length
        # sort = numpy.argsort(input_lengths)
        # inputs = inputs[sort]
        # targets = targets[sort]
        # input_lengths = input_lengths[sort]
        # target_lengths = target_lengths[sort]

        # Transpose data from example -> sequence to sequence -> example
        inputs = inputs.transpose(1, 0, 2)
        # Does each example have a target class (instead of an output vector)?
        classification = (len(targets.shape) == 2)
        if classification:
            targets = targets.transpose(1, 0)
        else:
            targets = targets.transpose(1, 0, 2)

        # Convert to theano shared vars
        self.inputs = theano.shared(inputs, borrow=True)
        self.targets = theano.shared(targets, borrow=True)
        self.input_lengths = theano.shared(input_lengths, borrow=True)
        self.target_lengths = theano.shared(target_lengths, borrow=True)

        # # Define a Theano shuffle function: shuffle inputs and targets in unison
        # # along the example dimension.
        # rng = T.shared_randomstreams.RandomStreams()
        # permutation = rng.permutation((), self.inputs.shape[1])
        # shuffled_inputs = self.inputs[:, permutation]
        # shuffled_targets = self.targets[:, permutation]

        # self.shuffle = theano.function([], [], updates=(
        #     (self.inputs, shuffled_inputs), 
        #     (self.targets, shuffled_targets)
        #     ))

        self.minibatch_size = minibatch_size
        self.minibatch_count = inputs.shape[1] / minibatch_size

        self.ttargets = ctc.transform_targets(self.targets)

    def minibatch(self, index):
        mb_start = self.minibatch_size * index
        mb_end = self.minibatch_size * (index + 1)

        input_lengths = T.cast(self.input_lengths[mb_start:mb_end], 'int32')
        target_lengths = T.cast(self.target_lengths[mb_start:mb_end], 'int32')

        inputs = self.inputs[:T.max(input_lengths), mb_start:mb_end]
        targets = self.targets[:T.max(target_lengths), mb_start:mb_end]
        
        return SequenceMinibatch(
            inputs,
            targets,
            input_lengths,
            target_lengths,
            ctc.transform_targets(targets)
        )