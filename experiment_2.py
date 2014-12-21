from input_layer import InputLayer
from tanh_layer import TanhLayer
from softmax_layer import SoftmaxLayer
from l2reg_layer import L2RegLayer

from sgd import SGD
from sgd_standard import Standard
from sgd_adadelta import Adadelta
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
learning_rule = Adadelta(decay = 0.90, epsilon = 0.0000001)
strategy = SGD(minibatch_size = 128, learning_rule=learning_rule)

# Initialize and run the training loop
train_loop(l2reg, strategy, training_data, validation_data, patience_factor=2, validation_frequency=10)