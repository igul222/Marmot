import numpy
import theano
import theano.tensor as T

from layer import Layer

class BatchNormalize(Layer):
    """Layer implementing batch normalization
       (see "Batch Normalization: Accelerating Deep Network Training by
       Reducing Internal Covariate Shift")."""

    # Numerical stability constant to prevent divide-by-zero
    # when normalizing variances.
    EPSILON = 0.00000001

    def __init__(self, prev_layer):
        super(BatchNormalize, self).__init__()

        self._prev_layer = prev_layer

        self.n_in = prev_layer.n_in
        self.n_out = prev_layer.n_out

        self._scale = theano.shared(value=numpy.float32(1))
        self._shift = theano.shared(value=numpy.float32(0))

        self.params = prev_layer.params + [self._scale, self._shift]

    def activations(self, dataset):
        prev_activations = self._prev_layer.activations(dataset)

        if prev_activations.ndim == 2:
            # flat dataset: (example, vector)
            mean = T.mean(prev_activations, axis=0)
            variance = T.var(prev_activations, axis=0)
        elif prev_activations.ndim == 3:
            # sequence dataset: (seq num, example, vector)
            mean = T.mean(prev_activations, axis=1).dimshuffle(0,'x',1)
            variance = T.var(prev_activations, axis=1).dimshuffle(0,'x',1)

        normalized = (prev_activations - mean) / T.sqrt(variance + self.EPSILON)
        scaled_and_shifted = (normalized * self._scale) + self._shift

        return scaled_and_shifted