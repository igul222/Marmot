import numpy
import theano
import theano.tensor as T

class Adadelta():

    def __init__(self, decay = 0.95, epsilon = 0.000001):
        self._decay = decay
        self._epsilon = epsilon

    def get_updates(self, param, grad):
        """
        Compute the AdaDelta updates
        Parameters
        ----------
        param : theano.tensor
        grad : theano.tensor
        """

        shape = param.get_value().shape
        mean_squared_grad = theano.shared(numpy.zeros(shape, dtype=theano.config.floatX), borrow=True)
        mean_squared_dx = theano.shared(numpy.zeros(shape, dtype=theano.config.floatX), borrow=True)

        # Accumulate gradient
        new_mean_squared_grad = (
            self._decay * mean_squared_grad +
            (1 - self._decay) * T.sqr(grad)
        )

        # Compute update
        old_rms_dx   = T.sqrt(mean_squared_dx + self._epsilon)
        new_rms_grad = T.sqrt(new_mean_squared_grad + self._epsilon)
        update = - old_rms_dx / new_rms_grad * grad

        # Accumulate updates
        new_mean_squared_dx = (
            self._decay * mean_squared_dx +
            (1 - self._decay) * T.sqr(update)
        )

        return [
            (mean_squared_grad, new_mean_squared_grad),
            (mean_squared_dx, new_mean_squared_dx),
            (param, param + update)
        ]