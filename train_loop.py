import numpy
import theano
import theano.tensor as T

import time

def _shared_dataset(dataset):
    """Load a dataset into Theano shared variables."""

    shared_x = theano.shared(numpy.asarray(dataset[0],
                                           dtype=theano.config.floatX),
                             borrow=True)

    shared_y = theano.shared(numpy.asarray(dataset[1],
                                           dtype=theano.config.floatX),
                             borrow=True)

    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue

    return (shared_x, T.cast(shared_y, 'int32'))

def train_loop(model,
               strategy,
               training_data,
               validation_data,
               patience_factor=2,
               validation_frequency=1):

    """Run to completion a training loop with early stopping."""

    training_data = _shared_dataset(training_data)
    validation_data = _shared_dataset(validation_data)

    train = strategy.training_function(model, training_data)

    calculate_training_accuracy = theano.function(
        inputs=[],
        outputs=model.accuracy(),
        givens={
          model.inputs:  training_data[0],
          model.targets: training_data[1]
          }
    )

    calculate_validation_accuracy = theano.function(
        inputs=[],
        outputs=model.accuracy(),
        givens={
          model.inputs:  validation_data[0],
          model.targets: validation_data[1]
          }
    )

    epoch = 0
    best_validation_accuracy = -numpy.inf
    best_validation_epoch = 0

    while epoch <= max(validation_frequency, best_validation_epoch * patience_factor):
        epoch += 1
        start_time = time.clock()

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
