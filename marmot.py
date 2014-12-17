from input_layer import InputLayer
from tanh_layer import TanhLayer
from softmax_layer import SoftmaxLayer
from l2reg_layer import L2RegLayer

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
hidden = TanhLayer(prev_layer=inputs, n=500)
softmax = SoftmaxLayer(prev_layer=hidden, n=10)
l2reg = L2RegLayer(prev_layer=softmax, reg_weight = 0.0001)

# Define a learning strategy
strategy = SGD(learning_rate = 0.01, minibatch_size = 20)

# Initialize and run the training loop
train_loop(l2reg, strategy, training_data, validation_data, patience=20, validation_frequency=10)