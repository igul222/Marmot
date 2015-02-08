import cPickle as pickle

class Layer(object):
    """Abstract layer base class."""

    def __init__(self):
        """Initialize the layer.
        Subclasses must provide a value for all attributes set to None here.
        """

        self.n_in = None
        self.n_out = None

        # self.inputs = None
        self.weight_params = []
        self.params = []
        # self.activations = None

    def dump_params(self):
        """Dump the params of this model (and all sub-models) to a binary
           string."""

        param_vals = [p.get_value(borrow=True) for p in self.params]
        return pickle.dumps(param_vals, pickle.HIGHEST_PROTOCOL)

    def load_params(self, param_dump):
        """Load params of this model (and all sub-models) from a binary string
           produced by dump_params."""

        new_params = pickle.loads(param_dump)
        for i in xrange(len(new_params)):
            self.params[i].set_value(new_params[i], borrow=True)