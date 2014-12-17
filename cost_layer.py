from layer import Layer

class CostLayer(Layer):
    """Abstract base class for layers which define cost functions.
       These layers are trainable and usually the last layer in a network.
    """

    def __init__(self):
        """Initialize the layer.

        Subclasses must provide a value for all attributes set to None here,
        as well as those in Layer.
        """
        super(CostLayer, self).__init__()

        self.targets = None

    def cost(self):
        raise NotImplemented

    def accuracy(self):
        raise NotImplemented