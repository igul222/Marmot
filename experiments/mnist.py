import marmot
import gzip
import cPickle

# Data file can be downloaded from: 
# http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
f = gzip.open('data/mnist.pkl.gz', 'rb')
training_data, validation_data, test_data = cPickle.load(f)
f.close()

# Load datasets onto the GPU
training_data = marmot.datasets.Simple(training_data[0], training_data[1], minibatch_size=128)
validation_data = marmot.datasets.Simple(validation_data[0], validation_data[1], minibatch_size=128)

# Build the model by composing layers
inputs  = marmot.layers.Input(28 * 28) # Each MNIST image has size 28*28
inputs  = marmot.layers.BatchNormalize(inputs)

hidden  = marmot.layers.Feedforward(prev_layer=inputs, n=500)
# hidden  = marmot.layers.BatchNormalize(hidden)

hidden  = marmot.layers.Feedforward(prev_layer=hidden, n=500)
# hidden  = marmot.layers.BatchNormalize(hidden)

hidden  = marmot.layers.Feedforward(prev_layer=hidden, n=500)
# hidden  = marmot.layers.BatchNormalize(hidden)

hidden  = marmot.layers.Feedforward(prev_layer=hidden, n=500)
# hidden  = marmot.layers.BatchNormalize(hidden)

softmax = marmot.layers.Softmax(prev_layer=hidden, n=10)
#l2reg   = marmot.layers.L2Reg(prev_layer=softmax, reg_weight = 1e-5)

# Define a learning strategy
learning_rule = marmot.sgd.Adadelta(decay = 0.75, epsilon = 1e-3)
strategy = marmot.sgd.SGD(learning_rule=learning_rule)

# Initialize and run the training loop
marmot.train_loop(
  softmax, 
  strategy, 
  training_data, 
  validation_data, 
  patience_factor=2, 
  validation_frequency=10
  )