class Standard():

  def __init__(self, learning_rate = 0.1):
    self._learning_rate = learning_rate

  def get_updates(self, param, grad):
    return [(param, param - self._learning_rate * grad)]