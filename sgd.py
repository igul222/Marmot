import theano
import theano.tensor as T


class SGD(object):

  def __init__(self,
               learning_rate=0.1,
               minibatch_size=128,
               ):
    self._learning_rate = learning_rate
    self._minibatch_size = minibatch_size

  def training_function(self, model, training_data):
    index = T.iscalar()
    
    cost = model.cost()

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