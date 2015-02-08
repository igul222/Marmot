import numpy
import theano

def pad(arrays, pad_val):
    """Pad each array in arrays with pad_val so that all arrays are of equal
       length, returning the new padded arrays and an array of their original 
       lengths.

       For multidimensional arrays, pad with an appropriately shaped array
       filled with pad_val."""

    lengths = [len(a) for a in arrays]
    max_len = max(lengths)

    def pad_array(arr):
        arr = numpy.asarray(arr)

        padding = numpy.full(
            (max_len - len(arr),) + arr.shape[1:],
            pad_val
        )
        return numpy.concatenate((arr, padding))

    return (
        numpy.array(map(pad_array, arrays), dtype=theano.config.floatX), 
        numpy.array(lengths)
        )

def parallel_shuffle(arrays):
    """Shuffle the given arrays in parallel. 
       Arrays must be of the same length."""

    indices = numpy.arange(len(arrays[0]))
    numpy.random.shuffle(indices)

    return [a[indices] for a in arrays]

def sort_by(sort_keys, arrays):
    """Sort the given arrays by the value of the corresponding entry in 
       sort_keys. Ex: sorting sequences by their lengths."""

    indices = numpy.argsort(sort_keys)
    return [a[indices] for a in arrays]
