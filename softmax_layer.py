from cost_layer import CostLayer

import numpy
import theano
import theano.tensor as T

class SoftmaxLayer(CostLayer):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, prev_layer, n):
        """Initialize the parameters of the logistic regression

        :type n: int
        :param n: number of units, the dimension of the space in
                      which the labels lie
        """
        super(SoftmaxLayer, self).__init__()

        self.n_in = prev_layer.n_in
        self.n_out = n

        # self.inputs = prev_layer.inputs
        self._prev_layer = prev_layer
        
        # Initialize the weights as a matrix of zeros with shape (n_in, n_out).
        self._weights = theano.shared(
            value=numpy.zeros(
                (prev_layer.n_out, self.n_out),
                dtype=theano.config.floatX
            ),
            name=self.uuid + '_weights',
            borrow=True
        )

        # Initialize the biases as a vector of n_out zeros.
        self._biases = theano.shared(
            value=numpy.zeros(
                (self.n_out,),
                dtype=theano.config.floatX
            ),
            name=self.uuid + '_biases',
            borrow=True
        )

        # Parameters of the model
        self.weight_params = prev_layer.weight_params + [self._weights]
        self.params = prev_layer.params + [self._weights, self._biases]

        # Targets variable that the model uses to calculate cost and accuracy;
        # for this model, a 1-dimensional vector of ints.
        # self.targets = T.ivector(self.uuid + '_targets')

        # Dot the inputs with the weights, add the biases, and apply softmax.
        # Softmax activations can be interpreted as P(y|x)
        # self.activations = T.nnet.softmax(T.dot(prev_layer.activations, weights) + biases)

        # Predicted value of y = the index of the P(y|x) vector whose value is maximal.
        # self._y_pred = T.argmax(self.activations, axis=1)

    def activations(self, inputs):
        # Dot the inputs with the weights, add the biases, and apply softmax.
        # Softmax activations can be interpreted as P(y|x)
        return T.nnet.softmax(
            T.dot(self._prev_layer.activations(inputs), self._weights) + self._biases
            )        

    def _y_pred(self, inputs):
        # Predicted value of y = the index of the P(y|x) vector whose value is maximal.
        return T.argmax(self.activations(inputs), axis=1)

    def cost(self, inputs, targets):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """

        # y.shape[0] is (symbolically) the number of rows in y, 
        # i.e. the number of examples in the minibatch.
        example_count = targets.shape[0]

        # T.arange(n) is a symbolic vector which will contain
        # [0,1,2,... n-1] 
        example_indices = T.arange(example_count)

        # T.log(self.activations) is a matrix of Log-Probabilities
        # with one row per example and one column per class.
        lp = T.log(self.activations(inputs))

        # lp[example_indices, y] is a vector containing the LP of
        # the correct label for each example in the minibatch.
        errors = lp[example_indices, targets]

        return -T.mean(errors)

    def accuracy(self, inputs, targets):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        return T.mean(T.eq(self._y_pred(inputs), targets))