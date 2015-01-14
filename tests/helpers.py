import numpy
import theano

def theano_eval(expr):
    """Evaluate a theano expression and return the result."""
    return theano.function([], expr)()

def assert_theano_equal(expr, target):
    """Assert that a Theano expression evaluates to a given result."""        
    result = theano_eval(expr)
    target = numpy.array(target)
    if not numpy.array_equal(result, target):
        numpy.set_printoptions(linewidth=319, suppress=True)
        raise AssertionError("Theano expression result:\n%s\nNot equal to target:\n%s" % (result, target))

def assert_theano_almost_equal(expr, target):
    """Assert that a Theano expression evaluates to a given result within a 
       small tolerance."""
    result = theano_eval(expr)
    target = numpy.array(target)
    if not numpy.allclose(result, target):
        numpy.set_printoptions(linewidth=319, suppress=True)
        raise AssertionError("Theano expression result:\n%s\nNot almost-equal to target:\n%s" % (result, target))