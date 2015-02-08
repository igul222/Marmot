# RNTN for sentence classification.
# Usage: python rntn.py [train|test] [model-file] input-file output-file
#
# For training, input should be a file with one labelled parse tree per line, 
# and output will be a dump of the trained model. For testing, you'll need to 
# specify a model file (the output of training), input should be a text file
# with one sentence per line, and output will be one label per line.

import numpy
import random
import math
import theano
import marmot
import sys
import cPickle as pickle

MINIBATCH_SIZE = 128
TRAINING_VALIDATION_SPLIT = 0.9

# Disable stdout buffering
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)
sys.stdout = Unbuffered(sys.stdout)

# Build the model
def build_model(wordmap, label_count):
    recursive = marmot.layers.Recursive(
        word_vec_length=30,
        wordmap=wordmap,
        tensor=True
    )
    softmax = marmot.layers.Softmax(prev_layer=recursive, n=label_count)
    return marmot.layers.L2Reg(prev_layer=softmax, reg_weight = 1e-6)

if sys.argv[1] == 'train':

    print "Loading data..."

    # Parse the file specified in the second command line arg
    with open(sys.argv[2]) as f:
        trees = [marmot.datasets.Trees.parse(l.strip('\n')) 
                 for l in f.readlines()]

    # Normally the Dataset class handles shuffling, but we need to do it now
    # because we want unbiased training/validation splits
    random.shuffle(trees)

    # Split into training/validation sets
    split = int(math.floor(len(trees) * TRAINING_VALIDATION_SPLIT))

    # Get the number of unique labels in the training set
    labels = set()
    def add_label(node):
        if node.is_leaf() and node.label not in labels:
            labels.add(node.label)
    for t in trees[:split]:
        t.traverse(add_label)
    label_count = len(labels)

    # Build datasets (this does a bunch of input preprocessing)
    training_data = marmot.datasets.Trees(trees[:split], minibatch_size=25)
    validation_data = marmot.datasets.Trees(
        trees[split:],
        wordmap=training_data.wordmap
    )

    learning_rule = marmot.sgd.Adadelta(decay = 0.90, epsilon = 1e-4)
    strategy = marmot.sgd.SGD(learning_rule=learning_rule)

    model = build_model(training_data.wordmap, label_count)

    print "Starting training..."

    # Train!
    results = marmot.train_loop(
        model,
        strategy,
        training_data,
        validation_data,
        patience_factor=2,
        validation_frequency=3,
    )

    # Save the best params to a file specified in the third command line arg
    model_data = {
        'param_dump': results['param_dump'],
        'label_count': label_count,
        'wordmap': training_data.wordmap
    }
    with open(sys.argv[3], 'w') as f:
        pickle.dump(model_data, f, pickle.HIGHEST_PROTOCOL)

elif sys.argv[1] == 'test':

    print "Loading data..."

    with open(sys.argv[2]) as f:
        model_data = pickle.load(f)

    with open(sys.argv[3]) as f:
        trees = [marmot.datasets.Trees.parse(l.strip('\n')) 
                 for l in f.readlines()]

    dataset = marmot.datasets.Trees(
        trees,
        wordmap=model_data['wordmap'],
        shuffle=False
    )

    model = build_model(model_data['wordmap'], model_data['label_count'])
    model.load_params(model_data['param_dump'])

    print "Predicting..."

    predictions = model.predictions(dataset).eval()
    outputs = []
    for t in xrange(len(trees)):
        node_strings = trees[t].traverse(str)
        for n in xrange(len(node_strings)):
            outputs.append(str(predictions[n][t]) + "\t" + node_strings[n])

    with open(sys.argv[4], 'w') as f:
        f.write("\n".join(outputs))

else:
    print "Please run with either train or test."