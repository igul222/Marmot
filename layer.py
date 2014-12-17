import uuid

class Layer(object):
    """Abstract layer base class."""

    def __init__(self):
        """Initialize the layer.
        Subclasses must provide a value for all attributes set to None here.
        """

        self.uuid = str(uuid.uuid4())

        self.n_in = None
        self.n_out = None

        self.inputs = None
        self.weight_params = None
        self.params = None
        self.activations = None