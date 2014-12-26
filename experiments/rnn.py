import marmot
import numpy

def make_data(delay, n):
    inputs = []
    targets = []
    
    for i in xrange(n):
        input = numpy.zeros((delay + 1, 10))
        target = numpy.zeros(delay + 1)

        hot_idx = numpy.random.randint(10)
        input[0][hot_idx] = 1
        target[-1] = hot_idx + 1

        inputs.append(input)
        targets.append(target)

    return (inputs, targets)

# Load datasets onto the GPU
training_data = make_data(delay=10, n=50000)
validation_data = make_data(delay=10, n=10000)
training_data = marmot.Dataset(training_data[0], training_data[1])
validation_data = marmot.Dataset(validation_data[0], validation_data[1])

# Build the model by composing layers
inputs  = marmot.layers.Input(10)
hidden  = marmot.layers.Recurrent(prev_layer=inputs, n=11)
softmax = marmot.layers.Softmax(prev_layer=hidden, n=11)
l2reg   = marmot.layers.L2Reg(prev_layer=softmax, reg_weight = 1e-3)

# Define a learning strategy
learning_rule = marmot.sgd.Adadelta(decay = 0.90, epsilon = 1e-6)
strategy = marmot.sgd.SGD(minibatch_size = 128, learning_rule=learning_rule)

# Initialize and run the training loop
marmot.train_loop(
    softmax, 
    strategy, 
    training_data, 
    validation_data, 
    patience_factor=2, 
    validation_frequency=10
    )