import theano
import theano.tensor as T
from sgd_standard import Standard

class SGD(object):

  def __init__(self,
               minibatch_size=128,
               learning_rule=Standard()
               ):
    self._learning_rule = learning_rule
    self._minibatch_size = minibatch_size

  def training_function(self, model, training_data):
    index = T.iscalar()
    
    cost = model.cost()

    updates = []
    for param in model.params:
      grad = T.grad(cost, wrt=param)
      param_updates = self._learning_rule.get_updates(param, grad)
      updates.extend(param_updates)

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