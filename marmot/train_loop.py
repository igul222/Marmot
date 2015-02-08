import numpy
import theano
import theano.tensor as T

import time

# KNOWN ISSUE: when computing mean cost / accuracy over minibatches, the last minibatch (which is smaller)
# is given the same weight as all the others -- which is bad. It's not a huge deal but should be eventually fixed.

# RELATED KNOWN ISSUE: In SGD the last batch will effectively be given more weight, for the same reason.
# This is mitigated by shuffling the training examples before each epoch, but still should be fixed eventually.

def train_loop(model,
               strategy,
               training_data,
               validation_data,
               min_patience=50,
               patience_factor=2,
               validation_frequency=3,
               silent=False):

    """Run to completion a training loop with early stopping."""

    train = strategy.training_function(model, training_data)

    calculate_training_accuracy = _accuracy_function(model, training_data)
    calculate_validation_accuracy = _accuracy_function(model, validation_data)

    epoch = 0
    best_validation_accuracy = -numpy.inf
    best_validation_epoch = 0
    best_param_dump = None

    while epoch < max(validation_frequency, min_patience, best_validation_epoch * patience_factor):
        epoch += 1
        start_time = time.time()

        # training_data.shuffle()
        mean_cost = train()

        if not silent:
            print 'epoch {0}: took {1}s, mean training cost {2}'.format(
                epoch,
                time.time() - start_time,
                mean_cost
                )

        if epoch % validation_frequency == 0:

            training_accuracy = calculate_training_accuracy()
            validation_accuracy = calculate_validation_accuracy()

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_validation_epoch = epoch
                best_param_dump = model.dump_params()

            if not silent:
                print 'epoch {0}: training accuracy {1}%, validation accuracy {2}% (best is {3}% from {4} epochs ago)'.format(
                    epoch,
                    training_accuracy * 100.,
                    validation_accuracy * 100.,
                    best_validation_accuracy * 100.,
                    epoch - best_validation_epoch
                    )

    return {
        'accuracy': best_validation_accuracy * 100.,
        'param_dump': best_param_dump
    }

def _accuracy_function(model, dataset):
    """
    Compile and return a function that calculates the given model's accuracy on
    the given dataset.

    (We do this calculation in minibatches to conserve memory.)
    """

    minibatch_index = T.iscalar()
    minibatch = dataset.minibatch(minibatch_index)

    mb_accuracy = theano.function(
        inputs=[minibatch_index],
        outputs=[model.accuracy(minibatch)]
    )

    def mean_accuracy():
        accuracies = [mb_accuracy(i) for i in xrange(dataset.minibatch_count)]
        return numpy.mean(accuracies)

    return mean_accuracy