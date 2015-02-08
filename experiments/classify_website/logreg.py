# Logistic regression for sentence classification.
# Usage: python logreg.py [train|test] [model-file] input-file output-file
#
# For training, input should be a TSV file with rows (label, sentence), and 
# output will be a dump of the trained model. For testing, you'll need to 
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
def build_model(wordmap, labelmap):
    inputs  = marmot.layers.Input(len(wordmap))
    return marmot.layers.Softmax(prev_layer=inputs, n=len(labelmap))

if sys.argv[1] == 'train':

    print "Loading data..."

    # Parse the TSV file specified in the second command line arg
    with open(sys.argv[2]) as f:
        data = [l.strip('\n').split('\t') for l in f.readlines()]

    # Normally the Dataset class handles shuffling, but we need to do it now
    # because we want unbiased training/validation splits
    random.shuffle(data)

    labels  = [d[0] for d in data]
    strings = [d[1] for d in data]

    # Split into training/validation sets
    split = int(math.floor(len(data) * TRAINING_VALIDATION_SPLIT))

    # Map each label class to an integer
    labelmap = marmot.Wordmap(words=labels[:split])
    labels = [labelmap.index_for_word(l) for l in labels]

    # Map each word to an int, and build bag-of-words vectors for each string.
    wordmap = marmot.Wordmap(strings=strings[:split])
    training_wordbags   = wordmap.to_wordbags(strings[:split])
    validation_wordbags = wordmap.to_wordbags(strings[split:])

    # Normalize feature vectors (TODO: Dataset class really should handle this)
    means = numpy.mean(training_wordbags, axis=0)
    stdevs = numpy.std(training_wordbags, axis=0)
    training_wordbags   = (training_wordbags   - means) / stdevs
    validation_wordbags = (validation_wordbags - means) / stdevs

    training_data = marmot.datasets.Simple(
        training_wordbags,
        labels[:split],
        minibatch_size=MINIBATCH_SIZE
    )
    validation_data = marmot.datasets.Simple(
        validation_wordbags, 
        labels[split:]
    )

    learning_rule = marmot.sgd.Adadelta(decay = 0.90, epsilon = 1e-4)
    strategy = marmot.sgd.SGD(learning_rule=learning_rule)

    model = build_model(wordmap, labelmap)

    print "Starting training..."

    # Train!
    results = marmot.train_loop(
        model,
        strategy,
        training_data,
        validation_data,
        min_patience=20,
        patience_factor=2,
        validation_frequency=1,
    )

    # Save the best params to a file specified in the third command line arg
    model_data = {
        'param_dump': results['param_dump'],
        'labelmap': labelmap,
        'wordmap': wordmap,
        'means': means,
        'stdevs': stdevs
    }
    with open(sys.argv[3], 'w') as f:
        pickle.dump(model_data, f, pickle.HIGHEST_PROTOCOL)

elif sys.argv[1] == 'test':

    print "Loading data..."

    with open(sys.argv[2]) as f:
        model_data = pickle.load(f)

    with open(sys.argv[3]) as f:
        sentences = [l.strip('\n') for l in f.readlines()]

    wordbags = model_data['wordmap'].to_wordbags(sentences)
    wordbags = (wordbags - model_data['means']) / model_data['stdevs']

    dataset = marmot.datasets.Simple(
        wordbags,
        numpy.zeros(len(sentences)), # the labels actually don't actually matter
        minibatch_size=MINIBATCH_SIZE,
        shuffle=False
    )

    model = build_model(model_data['wordmap'], model_data['labelmap'])
    model.load_params(model_data['param_dump'])

    print "Predicting..."

    pred_labels = [model_data['labelmap'].word_for_index(p) 
                   for p in model.predictions(dataset).eval()]

    output = [pred_labels[i] + "\t" + sentences[i] for i in xrange(len(sentences))]

    with open(sys.argv[4], 'w') as f:
        f.write("\n".join(output))

else:
    print "Please run with either train or test."