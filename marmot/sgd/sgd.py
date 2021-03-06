import numpy
import theano
import theano.tensor as T

from standard import Standard

class SGD(object):

    def __init__(self,
                 learning_rule=Standard(),
                 use_theano_scan=False
                 ):
        self._learning_rule = learning_rule
        self._use_theano_scan = use_theano_scan

    def training_function(self, model, training_data):

        def train_minibatch(index):
            minibatch = training_data.minibatch(index)

            cost = model.cost(minibatch)

            updates = []
            for param in model.params:
                grad = T.grad(cost, wrt=param)
                param_updates = self._learning_rule.get_updates(param, grad)
                updates.extend(param_updates)

            return cost, updates

        minibatch_count = training_data.minibatch_count

        # theano.scan is slower than a straight python loop (their fault,
        # not ours), but maybe one day it'll become faster, so it's left in here
        # as an option.
        if self._use_theano_scan:
            costs, updates = theano.scan(
                fn=train_minibatch,
                sequences=[T.cast(T.arange(minibatch_count), 'int32')]
                )

            return theano.function([], T.mean(costs), updates=updates)
        else:
            index = T.iscalar()
            cost, updates = train_minibatch(index)

            train_minibatch_fn = theano.function([index], cost, updates=updates)

            def train():
                costs = [train_minibatch_fn(i) for i in xrange(minibatch_count)]
                return numpy.mean(costs) # TODO investigate

            return train