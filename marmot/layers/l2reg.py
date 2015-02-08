from cost import Cost

class L2Reg(Cost):
    """Layer that provides L2 regularization on top of another cost layer."""

    def __init__(self, prev_layer, reg_weight=1):
        super(L2Reg, self).__init__()

        self._prev_layer = prev_layer
        self._reg_weight = reg_weight

        self.n_in = prev_layer.n_in
        self.n_out = prev_layer.n_out

        self.weight_params = prev_layer.weight_params
        self.params = prev_layer.params

    def cost(self, dataset):
        reg_term = sum([(w ** 2).sum() for w in self.weight_params])
        return self._prev_layer.cost(dataset) + (self._reg_weight * reg_term)

    def accuracy(self, dataset):
        return self._prev_layer.accuracy(dataset)

    def predictions(self, dataset):
        return self._prev_layer.predictions(dataset)