from cost import Cost

import numpy
import theano
import theano.tensor as T

class Softmax(Cost):
    """Softmax layer."""

    def __init__(self, prev_layer, n):
        super(Softmax, self).__init__()

        self.n_in = prev_layer.n_in
        self.n_out = n

        self._prev_layer = prev_layer
        
        # Initialize the weights as a matrix of zeros with shape (n_in, n_out).
        self._weights = theano.shared(
            value=numpy.zeros(
                (prev_layer.n_out, self.n_out),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # Initialize the biases as a vector of n_out zeros.
        self._biases = theano.shared(
            value=numpy.zeros(
                (self.n_out,),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # Parameters of the model
        self.weight_params = prev_layer.weight_params + [self._weights]
        self.params = prev_layer.params + [self._weights, self._biases]

    def activations(self, dataset):
        # We'll get around to implementing this if we find we ever actually
        # need it.
        raise NotImplemented

    def log_activations(self, dataset):
        # The reason we don't just do T.log(self.activations) is that 
        # (for reasons I don't understand) Theano's numerical stability 
        # optimization for log-softmax doesn't get applied, resulting in NaNs, 
        # which are no fun. Instead, we manually apply that optimization here.

        weighted_inputs = T.dot(
            self._prev_layer.activations(dataset),
            self._weights
        ) + self._biases

        def logsumexp(x, axis=None, keepdims=True):
            """A numerically stable version of log(sum(exp(x)))."""
            x_max = T.max(x, axis=axis, keepdims=True)
            preres = T.log(T.sum(T.exp(x - x_max),axis=axis,keepdims=keepdims))
            return preres + x_max

        # log_softmax(vector) = vector - logsumexp(vector)
        return weighted_inputs - logsumexp(
            weighted_inputs, 
            axis=weighted_inputs.ndim - 1
        )

    def predictions(self, dataset):
        log_activations = self.log_activations(dataset)
        return T.argmax(log_activations, axis=log_activations.ndim - 1)

    # TODO: write a test for this
    @staticmethod
    def _exclude_padding(matrix, lengths):
        """Given an (m, n) matrix of results and a length-m vector of sequence
           lengths, return all values in the matrix where n < lengths[m]. Use
           this to strip away padding outputs before calculating e.g. cost.
           """

        not_padding = (T.arange(matrix.shape[0]) \
                        .dimshuffle(0, 'x') \
                        .repeat(matrix.shape[1], axis=1)) < lengths

        # To prevent actual zero values from being ignored, we add an offset
        # before stripping nonzero values and then subtract that offset later.
        offset = 1 - matrix.min()
        return ((matrix + offset) * not_padding).nonzero_values() - offset

    def cost(self, dataset):
        targets = T.cast(dataset.targets, 'int32')

        # # Calculate log-probabilities at each timestep, example, and class.
        lp = self.log_activations(dataset)

        if lp.ndim == 3:
            # Theano's advanced indexing is limited, so we reshape our 
            # (n_steps, n_seq, n_classes) tensor of probs to a 
            # (n_steps * n_seq, n_classes) matrix, pluck the target activations,
            # and reshape back.
            flat_lp = T.reshape(lp, (lp.shape[0] * lp.shape[1], -1))
            flat_targets = targets.flatten(ndim=1)
            flat_errors = -flat_lp[T.arange(flat_lp.shape[0]), flat_targets]
            errors = flat_errors.reshape((lp.shape[0], lp.shape[1]))
            return T.mean(self._exclude_padding(errors, dataset.lengths))
        else:
            errors = -lp[T.arange(lp.shape[0]), targets]
            return T.mean(errors)

    def accuracy(self, dataset):
        targets = T.cast(dataset.targets, 'int32')
        results = T.eq(self.predictions(dataset), targets)
        if results.ndim == 2:
            return T.mean(self._exclude_padding(results, dataset.lengths))
        else:
            return T.mean(results)