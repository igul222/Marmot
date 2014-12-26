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
               validation_frequency=1):

    """Run to completion a training loop with early stopping."""

    train = strategy.training_function(model, training_data)

    calculate_accuracy = theano.function(
        inputs=[],
        outputs=[
          model.accuracy(training_data.inputs, training_data.targets),
          model.accuracy(validation_data.inputs, validation_data.targets)
          ]
    )

    epoch = 0
    best_validation_accuracy = -numpy.inf
    best_validation_epoch = 0

    while epoch < max(validation_frequency, min_patience, best_validation_epoch * patience_factor):
        epoch += 1
        start_time = time.time()

        training_data.shuffle()
        mean_cost = train()

        print 'epoch {0}: took {1}s, mean training cost {2}'.format(
            epoch,
            time.time() - start_time,
            mean_cost
            )

        if epoch % validation_frequency == 0:

            training_accuracy, validation_accuracy = calculate_accuracy()

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_validation_epoch = epoch

            print 'epoch {0}: training accuracy {1}%, validation accuracy {2}% (best is {3}% from {4} epochs ago)'.format(
                epoch,
                training_accuracy * 100.,
                validation_accuracy * 100.,
                best_validation_accuracy * 100.,
                epoch - best_validation_epoch
                )