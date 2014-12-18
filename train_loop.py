from dataset import Dataset

import numpy
import theano
import theano.tensor as T

import time

def train_loop(model,
               strategy,
               training_data,
               validation_data,
               patience_factor=2,
               validation_frequency=1):

    """Run to completion a training loop with early stopping."""

    training_data = Dataset(training_data)
    validation_data = Dataset(validation_data)

    train = strategy.training_function(model, training_data)

    calculate_training_accuracy = theano.function(
        inputs=[],
        outputs=model.accuracy(),
        givens={
          model.inputs:  training_data.inputs,
          model.targets: training_data.targets
          }
    )

    calculate_validation_accuracy = theano.function(
        inputs=[],
        outputs=model.accuracy(),
        givens={
          model.inputs:  validation_data.inputs,
          model.targets: validation_data.targets
          }
    )

    epoch = 0
    best_validation_accuracy = -numpy.inf
    best_validation_epoch = 0

    while epoch <= max(validation_frequency, best_validation_epoch * patience_factor):
        epoch += 1
        start_time = time.clock()

        training_data.shuffle()
        costs = train()

        print 'epoch {0}: took {1}s, mean training cost {2}'.format(
            epoch,
            time.clock() - start_time,
            numpy.mean(costs)
            )

        if epoch % validation_frequency == 0:

            training_accuracy = calculate_training_accuracy()
            validation_accuracy = calculate_validation_accuracy()

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