import theano
import theano.tensor as T


inputs = T.tensor3()
weights = T.vector()
biases = T.scalar()

activations = T.dot(inputs, weights) + biases

run = theano.function([inputs, weights, biases], activations)

_inputs = [[[1,2,3],[4,5,6]], [[1,2,3],[4,5,6]]]
_weights = [0,1,0]
_biases = 10

print run(_inputs, _weights, _biases)