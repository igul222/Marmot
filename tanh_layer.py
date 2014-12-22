from layer import Layer

import numpy
import theano
import theano.tensor as T

class TanhLayer(Layer):
    """Feedforward layer with tanh activation function."""

    def __init__(self, prev_layer, n):
        """Initialize the layer.
        Subclasses must provide a value for all attributes set to None here.
        """
        super(TanhLayer, self).__init__()

        rng = numpy.random.RandomState()

        self.n_in = prev_layer.n_in
        self.n_out = n

        # self.inputs = prev_layer.inputs

        self._prev_layer = prev_layer

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
            name=self.uuid + '_weights',
            borrow=True
            )

        bias_values = numpy.zeros((self.n_out,), dtype=theano.config.floatX)
        self._biases = theano.shared(
            value=bias_values,
            name=self.uuid + '_biases',
            borrow=True
            )

        self.weight_params = prev_layer.weight_params + [self._weights]
        self.params = prev_layer.params + [self._weights, self._biases]

        # self.activations = T.tanh(
        #     T.dot(prev_layer.activations, weights) + biases
        #     )

    def activations(self, inputs):
        return T.tanh(
            T.dot(self._prev_layer.activations(inputs), self._weights) + self._biases
            )