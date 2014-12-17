from input_layer import InputLayer
from softmax_layer import SoftmaxLayer
from sgd import SGD
from train_loop import train_loop

import gzip
import cPickle

# Data file can be downloaded from: 
# http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
f = gzip.open('data/mnist.pkl.gz', 'rb')
training_data, validation_data, test_data = cPickle.load(f)
f.close()

# Build the model by composing layers
inputs = InputLayer(28 * 28) # Each MNIST image has size 28*28
model = SoftmaxLayer(prev_layer=inputs, n_out=10)

# Define a learning strategy
strategy = SGD(learning_rate = 0.13, minibatch_size = 600)

# Initialize and run the training loop
train_loop(model, strategy, training_data, validation_data, patience=20, validation_frequency=10)