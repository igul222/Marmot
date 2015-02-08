import numpy
import theano
import theano.tensor as T

from layer import Layer

class Feedforward(Layer):
    """Feedforward layer with tanh activation function."""

    def __init__(self, prev_layer, n, activation_fn = T.tanh):
        """Initialize the layer.
        Subclasses must provide a value for all attributes set to None here.
        """
        super(Feedforward, self).__init__()

        self._prev_layer = prev_layer
        self._activation_fn = activation_fn

        self.n_in = prev_layer.n_in
        self.n_out = n

        rng = numpy.random.RandomState()

        weight_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6. / (prev_layer.n_out + self.n_out)),
                high=numpy.sqrt(6. / (prev_layer.n_out + self.n_out)),
                size=(prev_layer.n_out, self.n_out)
                ),
            dtype=theano.config.floatX
            )
        self._weights = theano.shared(
            value=weight_values, 
            borrow=True
            )

        bias_values = numpy.zeros((self.n_out,), dtype=theano.config.floatX)
        self._biases = theano.shared(
            value=bias_values,
            borrow=True
            )

        self.weight_params = prev_layer.weight_params + [self._weights]
        self.params = prev_layer.params + [self._weights, self._biases]

    def activations(self, dataset):
        return self._activation_fn(
            T.dot(
                self._prev_layer.activations(dataset), 
                self._weights) + self._biases
        )