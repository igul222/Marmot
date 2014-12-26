from cost import Cost

class L2Reg(Cost):
    """Layer that provides L2 regularization on top of another cost layer."""

    def __init__(self, prev_layer, reg_weight=1):
        super(L2Reg, self).__init__()

        self._prev_layer = prev_layer
        self._reg_weight = reg_weight

        self.n_in = prev_layer.n_in
        self.n_out = prev_layer.n_out

        # self.inputs = prev_layer.inputs
        self.weight_params = prev_layer.weight_params
        self.params = prev_layer.params
        # self.activations = prev_layer.activations

        # self.targets = prev_layer.targets

    def activations(self, inputs):
        return self._prev_layer.activations(inputs)

    def cost(self, inputs, targets):
        reg_term = sum([(w ** 2).sum() for w in self.weight_params])
        return self._prev_layer.cost(inputs, targets) + (self._reg_weight * reg_term)

    def accuracy(self, inputs, targets):
        return self._prev_layer.accuracy(inputs, targets)