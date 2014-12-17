import numpy
import theano
import theano.tensor as T
import time

N = 40000
feats = 784
D = (
  numpy.random.randn(N, feats).astype(theano.config.floatX), 
  numpy.random.randint(size=N, low=0, high=2).astype(theano.config.floatX)
  )
training_steps = 100

# Load data onto the GPU
x = theano.shared(D[0], name='x')
y = theano.shared(D[1], name='y')

# Declare Theano symbolic variables
# x = T.matrix("x")
# y = T.vector("y")
w = theano.shared(numpy.random.randn(feats).astype(theano.config.floatX), name="w")
b = theano.shared(numpy.asarray(0., dtype=theano.config.floatX), name="b")


# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = T.cast(xent.mean(), theano.config.floatX) + 0.01 * (w ** 2).sum() # The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # (we shall return to this in a
                                          # following section of this tutorial)

# Compile
train = theano.function(
          inputs=[],
          outputs=[],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb))
          )

errors = prediction - y
predict = theano.function(inputs=[], outputs=errors)


print "Training..."

t0 = time.time()

# Train
for i in range(training_steps):
  train()

t1 = time.time()


print predict()
print "Finished in %f sec" % (t1 - t0)