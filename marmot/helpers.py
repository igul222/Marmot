import numpy
import theano
import theano.tensor as T

# TODO cleanup these two functions (probably can be merged into one?)

def right_shift(x, shift, pad_val=numpy.float32(0)):
    """Right-shift along the first dimension of a matrix, padding the left with
       a given value."""
    return T.concatenate([
            T.alloc(pad_val, shift, x.shape[1]),
            x[:-shift]
        ], axis=0)

def right_shift_rows(x, shift, pad_val=numpy.float32(0)):
    """Right-shift rows in a matrix, padding the left with a given value."""
    return T.concatenate([
            T.alloc(pad_val, x.shape[0], shift),
            x[:,:-shift]
        ], axis=1)