import numpy
import theano
import theano.tensor as T

from input import Input
from softmax import Softmax
from cost import Cost

class Recursive(Cost):
    """Recursive neural network model. Operates on tree datasets."""

    def __init__(self, word_vec_length, wordmap, tensor=True):
        super(Recursive, self).__init__()

        self.tensor = tensor
        self._word_vec_length = word_vec_length
        self.n_out = word_vec_length

        # Initialize the word vectors to random values in U(-0.0001, 0.0001).
        L = numpy.random.uniform(
            low=-0.0001,
            high=0.0001,
            size=(len(wordmap), word_vec_length)
        )
        self.L = theano.shared(L.astype(theano.config.floatX), borrow=True)

        if tensor:
            V = numpy.random.uniform(
                low=-0.01,
                high=0.01,
                size=(2*word_vec_length, 2*word_vec_length, word_vec_length)
            )
            self.V = theano.shared(V.astype(theano.config.floatX), borrow=True)

        # Initialize the weights to two concatenated identity matrices plus a
        # small amount of noise (not in the paper but seemed reasonable).
        eye = numpy.eye(word_vec_length)
        W = 0.5 * numpy.concatenate((eye, eye), axis=0)
        W += numpy.random.uniform(
            low=-0.01,
            high=0.01,
            size=(2*word_vec_length, word_vec_length)
        )
        self.W = theano.shared(W.astype(theano.config.floatX), borrow=True)

        # Initialize biases to zeros
        # (no good justification for this)
        b = numpy.random.uniform(
            low=-0.00,
            high=0.00,
            size=word_vec_length
        )
        self.b = theano.shared(b.astype(theano.config.floatX), borrow=True)

        # TODO: weight_params is really ugly; fix.
        if self.tensor:
            self.weight_params = [self.V, self.W]
        else:
            self.weight_params = [self.W]
        self.params = [self.L, self.b] + self.weight_params

    def activations(self, dataset):
        n_steps = dataset.is_leafs.shape[0]
        n_examples = dataset.is_leafs.shape[1]
        
        def step(i, is_leaf, word_index, child_index, activations):
            # If leaf node, grab the word vector from the lookup table
            leaves = self.L[word_index]

            # Otherwise, compose the left and right child vectors
            lefts = activations[child_index[0], T.arange(n_examples)]
            rights = activations[child_index[1], T.arange(n_examples)]
            stacked = T.concatenate((lefts, rights), axis=1)
            if self.tensor:
                tdot = T.tensordot(stacked, self.V, [[1], [0]])

                # batched_dot is really slow because it doesn't process
                # examples in parallel. A simple elementwise multiply and reduce
                # works a whole lot better.
                # tdot = T.batched_dot(stacked, tdot)
                tdot = (stacked.dimshuffle(0, 1, 'x') * self.W).sum(axis=1)

                composites = T.tanh(tdot + T.dot(stacked, self.W) + self.b)
            else:
                composites = T.tanh(T.dot(stacked, self.W) + self.b)

            result = T.switch(T.shape_padright(is_leaf, 1), leaves, composites)

            return T.set_subtensor(activations[i], result)

        blank_activations = T.alloc(
            numpy.float32(0),
            n_steps,
            n_examples,
            self._word_vec_length)

        activations_all_timesteps, _ = theano.scan(
            step,
            sequences=(
                T.arange(n_steps),
                dataset.is_leafs, 
                T.cast(dataset.word_indices, 'int32'), 
                T.cast(dataset.child_indices, 'int32')
            ),
            outputs_info=blank_activations
        )

        return activations_all_timesteps[-1]
