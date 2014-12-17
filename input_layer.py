from layer import Layer

import theano.tensor as T

class InputLayer(Layer):

  def __init__(self, n):
    """Initialize the input layer.

    :type n: int
    :param n: number input units; the dimension of the space in which the
              inputs lie.
    """
    super(InputLayer, self).__init__()

    self.n_in = n
    self.n_out = n

    self.inputs = T.matrix(self.uuid + '_inputs')
    self.activations = self.inputs

    self.params = []