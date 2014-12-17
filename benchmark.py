import numpy
import theano
import theano.tensor as T
import time

minibatch_count = 100
minibatch_size = 60
layer_count = 3
layer_size = 500

weight_matrices = []
for i in range(layer_count):
  weights = numpy.random.randn(layer_size, layer_size).astype(theano.config.floatX)
  weight_matrices.append(theano.shared(weights))

inputs = theano.shared(numpy.zeros((layer_size, minibatch_size)).astype(theano.config.floatX), 'inputs')

activations = inputs
for i in range(layer_count):
  activations = T.dot(weight_matrices[i], activations)

network = theano.function([], [activations])

t0 = time.time()

for i in range(minibatch_count):
  inputs.set_value(numpy.random.randn(layer_size, minibatch_size).astype(theano.config.floatX))
  network()

t1 = time.time()

print "Finished in %f sec" % (t1 - t0)