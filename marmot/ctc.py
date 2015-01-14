import numpy
import theano
import theano.tensor as T

# log(0) = -infinity, but this leads to
# NaN errors in log_add and elsewhere,
# so we'll just use a large negative value.
_LOG_ZERO = numpy.float32(-100000)
_LOG_ONE = numpy.float32(0)

# Index of the output neuron corresponding to a "blank" prediction
_BLANK = T.cast(numpy.float32(0), 'int32')

def _log_add(log_a, log_b):
    """Theano expression for log(a+b) given log(a) and log(b)."""
    smaller = T.minimum(log_a, log_b)
    larger = T.maximum(log_a, log_b)
    return larger + T.log1p(T.exp(smaller - larger))

def _log_add_3(log_a, log_b, log_c):
    """Theano expression for log(a+b+c) given log(a), log(b), log(c)."""
    smaller = T.minimum(log_a, log_b)
    larger = T.maximum(log_a, log_b)
    largest = T.maximum(larger, log_c)
    larger = T.minimum(larger, log_c)

    return largest + T.log1p(
            T.exp(smaller - largest) + 
            T.exp(larger - largest)
            )

def _right_shift(x, shift):
    """Right-shift rows in a matrix, padding the left with log(0)."""
    return T.concatenate([
            T.alloc(_LOG_ZERO, x.shape[0], shift),
            x[:,:-shift]
        ], axis=1)

def _initial_probabilities(example_count, target_length):
    """The initial value of the forward-backward
       recurrence: a matrix of shape (example_count, target_length) where
       each row is log([1,0,0,...])."""

    row = T.concatenate([
        T.alloc(_LOG_ONE, 1),
        T.alloc(_LOG_ZERO, target_length - 1)
    ])

    return T.shape_padleft(row).repeat(example_count, axis=0)

def _skip_allowed(targets):
    """A matrix of shape (example_count, target_length). For each example, 
       values are log(1) if a transition is allowed from a given label to 
       the next-to-next label (a "skip"), and log(0) otherwise. 

       Skip conditions:
           a) next label is blank, and
           b) the next to next label is different from the current

       * note that (b) implies (a), so we really only need to check (b).
    """

    targets = T.concatenate([
            targets,
            T.shape_padleft([_BLANK, _BLANK]).repeat(targets.shape[0], axis=0)
        ], axis=1)
    skip_allowed = T.neq(targets[:, :-2], targets[:, 2:])# * T.eq(y_extend[:, 1:-1], blank)

    # Since the values of skip_allowed are all 1 or 0, we apply a linear 
    # function that maps 1 to log(1) and 0 to log(0). This lets us use our 
    # "special" val of log(0) and might just be faster than doing 
    # T.log(skip_allowed).
    return (skip_allowed * (_LOG_ONE - _LOG_ZERO)) + _LOG_ZERO

def _forward_vars(activations, targets):
    """Calculate the CTC forward variables: for each example, a matrix of 
       shape (sequence length, target length) where entry (t,u) corresponds 
       to the log-probability of the network predicting the target sequence 
       prefix [0:u] by time t."""

    targets = T.cast(targets, 'int32')

    activations = T.log(activations)

    # For each example, a matrix of shape (seq len, target len) with values
    # corresponding to activations at each time and sequence position.
    probs = activations[:, T.shape_padleft(T.arange(activations.shape[1])).T, targets]

    initial_probs = _initial_probabilities(
        probs.shape[1], 
        targets.shape[1])

    skip_allowed = _skip_allowed(targets)

    def step(p_curr, p_prev):
        no_change = p_prev
        next_label = _right_shift(p_prev, 1)
        skip = _right_shift(p_prev + skip_allowed, 2)

        return p_curr + _log_add_3(no_change, next_label, skip)

    probabilities, _ = theano.scan(
        step,
        sequences=[probs],
        outputs_info=[initial_probs]
    )

    return probabilities

def cost(activations, targets):
    """Calculate the CTC cost: the mean of the negative log-probabilities of 
       the correct labellings for each example."""
    forward_vars = _forward_vars(activations, targets)
    return -T.mean(_log_add(forward_vars[-1,:,-1], forward_vars[-1,:,-2]))

def _best_path_decode(activations):
    """Calculate the CTC best-path decoding for a given activation sequence.
       In the returned matrix, shorter sequences are padded with -1s."""

    # For each timestep, get the highest output
    decoding = T.argmax(activations, axis=2)

    # prev_outputs[time][example] == decoding[time - 1][example]
    prev_outputs = T.concatenate([T.alloc(_BLANK, 1, decoding.shape[1]), decoding], axis=0)[:-1]

    # Filter all repetitions to zero (blanks are already zero)
    decoding = decoding * T.neq(decoding, prev_outputs)

    # Calculate how many blanks each sequence has relative to longest sequence
    blank_counts = T.eq(decoding, 0).sum(axis=0)
    min_blank_count = T.min(blank_counts)
    max_seq_length = decoding.shape[0] - min_blank_count # used later
    padding_needed = blank_counts - min_blank_count

    # Generate the padding matrix by ... doing tricky things
    max_padding_needed = T.max(padding_needed)
    padding_needed = padding_needed.dimshuffle('x',0).repeat(max_padding_needed, axis=0)
    padding = T.arange(max_padding_needed).dimshuffle(0,'x').repeat(decoding.shape[1],axis=1)
    # We pad with -1 because it's nonzero but still not a valid label
    padding = -1 * T.lt(padding, padding_needed)

    # Apply the padding
    decoding = T.concatenate([decoding, padding], axis=0)

    # Remove zero values
    nonzero_vals = decoding.T.nonzero_values()
    decoding = T.reshape(nonzero_vals, (decoding.shape[1], max_seq_length)).T

    return decoding

# def accuracy(activations, targets, blank):
#     best_paths = T.argmax(activations, axis=2)