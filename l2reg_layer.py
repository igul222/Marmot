from cost_layer import CostLayer

class L2RegLayer(CostLayer):
    """Layer that provides L2 regularization on top of another cost layer."""

    def __init__(self, prev_layer, reg_weight=1):
        super(CostLayer, self).__init__()

        self._prev_layer = prev_layer
        self._reg_weight = reg_weight

        self.n_in = prev_layer.n_in
        self.n_out = prev_layer.n_out

        self.inputs = prev_layer.inputs
        self.weight_params = prev_layer.weight_params
        self.params = prev_layer.params
        self.activations = prev_layer.activations

        self.targets = prev_layer.targets


    def cost(self):
        reg_term = sum([(w ** 2).sum() for w in self.weight_params])
        return self._prev_layer.cost() + (self._reg_weight * reg_term)

    def accuracy(self):
        return self._prev_layer.accuracy()