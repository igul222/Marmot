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
























import theano
import theano.tensor as T

class SGDAdadelta(object):

  def __init__(self,
               minibatch_size=128,
               decay=0.95
               ):
    self._minibatch_size = minibatch_size
    self._decay = decay

  def training_function(self, model, training_data):
    index = T.iscalar()
    
    cost = model.cost()

    param_shapes = [p.get_value().shape for p in model.params]

    mean_squared_grad = [theano.shared(np.zeros(shape), borrow=True) 
                        for shape in param_shapes]

    mean_squared_dx   = [theano.shared(np.zeros(shape), borrow=True) 
                        for shape in param_shapes]

    new_mean_squared_grad = (
      self._decay * mean_squared_grad +
      (1 - self._decay) * T.sqr()
      )

    [ rho*g_sq + (1-rho)*(g**2) for g_sq,g in izip(gradients_sq,gradients) ]


    updates = [
      ( param, param - self._learning_rate * T.grad(cost, wrt=param) )
      for param in model.params
      ]

    train_minibatch = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
          model.inputs:  training_data.inputs[index * self._minibatch_size : (index + 1) * self._minibatch_size],
          model.targets: training_data.targets[index * self._minibatch_size : (index + 1) * self._minibatch_size]
          }
        )

    minibatch_count = training_data.example_count / self._minibatch_size

    def train():
      return [train_minibatch(i) for i in xrange(minibatch_count)]

    return train


def updates(parameters,gradients,rho,eps):
  # create variables to store intermediate updates
  gradients_sq = [ U.create_shared(np.zeros(p.get_value().shape)) for p in parameters ]
  deltas_sq = [ U.create_shared(np.zeros(p.get_value().shape)) for p in parameters ]
 
  # calculates the new "average" delta for the next iteration
  gradients_sq_new = [ rho*g_sq + (1-rho)*(g**2) for g_sq,g in izip(gradients_sq,gradients) ]
 
  # calculates the step in direction. The square root is an approximation to getting the RMS for the average value
  deltas = [ (T.sqrt(d_sq+eps)/T.sqrt(g_sq+eps))*grad for d_sq,g_sq,grad in izip(deltas_sq,gradients_sq_new,gradients) ]
 
  # calculates the new "average" deltas for the next step.
  deltas_sq_new = [ rho*d_sq + (1-rho)*(d**2) for d_sq,d in izip(deltas_sq,deltas) ]
 
  # Prepare it as a list f
  gradient_sq_updates = zip(gradients_sq,gradients_sq_new)
  deltas_sq_updates = zip(deltas_sq,deltas_sq_new)
  parameters_updates = [ (p,p - d) for p,d in izip(parameters,deltas) ]
  return gradient_sq_updates + deltas_sq_updates + parameters_updates