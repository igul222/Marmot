# # TODO: right now this is just a copy of the old sequences.py
# # turn it into a proper subclass of sequences

# import numpy
# import theano
# import theano.tensor as T

# import marmot.ctc as ctc
# import helpers

# class SequenceMinibatch(object):
#     def __init__(self, inputs, targets, input_lengths, target_lengths, ttargets):
#         self.inputs = inputs
#         self.targets = targets
#         self.input_lengths = input_lengths
#         self.target_lengths = target_lengths
#         self.ttargets = ttargets

# class Sequences(object):

#     def __init__(
#         self, 
#         inputs, 
#         targets, 
#         minibatch_size=128,
#         input_padding_val=0,
#         target_padding_val=ctc.PADDING
#         ):
#         """Load a dataset into Theano shared variables."""

#         # Apply padding
#         inputs, input_lengths = helpers.pad(inputs, input_padding_val)
#         targets, target_lengths = helpers.pad(targets, target_padding_val)

#         # Shuffle
#         inputs, \
#         targets, \
#         input_lengths, \
#         target_lengths = helpers.parallel_shuffle(
#             (inputs, targets, input_lengths, target_lengths)
#         )

#         # Now sort in order of increasing input length
#         inputs, \
#         targets, \
#         input_lengths, \
#         target_lengths = helpers.sort_by(
#             input_lengths,
#             (inputs, targets, input_lengths, target_lengths)
#         )

#         # Transpose data from example -> sequence to sequence -> example
#         inputs = inputs.transpose(1, 0, 2)
#         # Does each example have a target class (instead of an output vector)?
#         classification = (len(targets.shape) == 2)
#         if classification:
#             targets = targets.transpose(1, 0)
#         else:
#             targets = targets.transpose(1, 0, 2)

#         # Load into theano shared vars
#         self.inputs = theano.shared(inputs, borrow=True)
#         self.targets = theano.shared(targets, borrow=True)
#         # these need to be stored on the GPU as floats and casted when needed
#         self.input_lengths = theano.shared(
#             input_lengths.astype(theano.config.floatX), 
#             borrow=True)
#         self.target_lengths = theano.shared(
#             target_lengths.astype(theano.config.floatX), 
#             borrow=True)

#         self.minibatch_size = minibatch_size
#         self.minibatch_count = inputs.shape[1] / minibatch_size

#         self.ttargets = ctc.transform_targets(self.targets)

#     def minibatch(self, index):
#         mb_start = self.minibatch_size * index
#         mb_end = self.minibatch_size * (index + 1)

#         input_lengths = T.cast(self.input_lengths[mb_start:mb_end], 'int32')
#         target_lengths = T.cast(self.target_lengths[mb_start:mb_end], 'int32')

#         inputs = self.inputs[:T.max(input_lengths), mb_start:mb_end]
#         targets = self.targets[:T.max(target_lengths), mb_start:mb_end]
        
#         return SequenceMinibatch(
#             inputs,
#             targets,
#             input_lengths,
#             target_lengths,
#             ctc.transform_targets(targets)
#         )