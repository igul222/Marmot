import numpy
import theano
import theano.tensor as T

class Dataset(object):

    def __init__(self, dataset):
        """Load a dataset into Theano shared variables."""

        self.example_count = len(dataset[0])

        inputs = theano.shared(numpy.asarray(dataset[0],
                                                  dtype=theano.config.floatX),
                                    borrow=True)

        self._targets = theano.shared(numpy.asarray(dataset[1],
                                              dtype=theano.config.floatX),
                                borrow=True)

        self.inputs = inputs
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue    
        # TODO fix this ugly piece of shit
        self.targets = T.cast(self._targets, 'int32')

    def shuffle(self):
        pass
        inputs = self.inputs.get_value()
        targets = self._targets.get_value()

        # Shuffle inputs and targets in unison
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(inputs)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)

        self.inputs.set_value(inputs)
        self._targets.set_value(targets)
