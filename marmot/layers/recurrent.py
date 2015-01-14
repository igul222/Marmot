import numpy
import theano
import theano.tensor as T

from layer import Layer

class Recurrent(Layer):
    """Abstract layer base class."""

    def __init__(self, prev_layer, n):
        super(Recurrent, self).__init__()
        
        self.n_in = prev_layer.n_in
        self.n_out = n

        self._prev_layer = prev_layer

        # Initialize incoming weights
        input_weight_values = numpy.asarray(
            numpy.random.uniform(size=(prev_layer.n_out, n),
                                low=-.01, 
                                high=.01),
            dtype=theano.config.floatX)

        self._input_weights = theano.shared(value=input_weight_values, borrow=True)

        # Initialize recurrent weights
        recurrent_weight_values = numpy.asarray(
            numpy.random.uniform(size=(n, n),
                                low=-.01, 
                                high=.01),
            dtype=theano.config.floatX)

        self._recurrent_weights = theano.shared(value=recurrent_weight_values, borrow=True)

        # Initialize h0 (initial hidden state)
        h0_values = numpy.zeros((n,), dtype=theano.config.floatX)
        self._h0 = theano.shared(value=h0_values)

        # Initialize biases
        bias_values = numpy.zeros((n,), dtype=theano.config.floatX)
        self._biases = theano.shared(value=bias_values)

        self.weight_params = [self._input_weights, self._recurrent_weights]
        self.params = self.weight_params + [self._h0, self._biases]

    def activations(self, inputs):
        prev_layer_activations = self._prev_layer.activations(inputs)
        biased_weighted_inputs = T.dot(prev_layer_activations, self._input_weights) + self._biases

        def step(current_biased_weighted_inputs, last_activations):
            return T.tanh(
                current_biased_weighted_inputs +
                T.dot(last_activations, self._recurrent_weights)
            )

        activations, updates = theano.scan(
            step,
            sequences=biased_weighted_inputs,
            outputs_info=T.alloc(self._h0, inputs.shape[1], self.n_out)
            )

        return activations