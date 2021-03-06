import numpy
import theano
import theano.tensor as T

import helpers

def distances(a, b, padding_val):
    a = T.as_tensor_variable(a)
    b = T.as_tensor_variable(b)

    n_strings = a.shape[1]
    i_max = b.shape[0] + a.shape[0]
    j_max = a.shape[0]

    def step(i, prev, prev_prev):
        j = T.arange(j_max + 1)

        x = (i - j) % (b.shape[0] + 1)
        y = j

        # The main rule for filling the matrix:
        # If current a == current b, keep the val from (x-1,y-1).
        # Otherwise new val = min(upper, left, upper-left) + 1.
        new_vals = T.switch(
            T.eq(b[x-1], a[y-1]),
            helpers.right_shift(prev_prev, 1),
            T.min([
                prev,
                helpers.right_shift(prev, 1),
                helpers.right_shift(prev_prev, 1),
            ], axis=0) + 1
        )

        # ... but if current a == padding, keep the val from (x, y-1)
        new_vals = T.switch(
            T.eq(a[y-1], padding_val),
            helpers.right_shift(prev, 1),
            new_vals
        )
        # ... likewise if current b == padding, keep the val from (x-1, y)
        new_vals = T.switch(
            T.eq(b[x-1], padding_val),
            prev,
            new_vals
        )
        # (If they're both padding, it doesn't matter which val we keep.)

        # Finally, if the value corresponds to an initial value of the matrix,
        # use that instead of anything above.
        is_initial = T.any([T.eq(j, 0), T.eq(j, i)], axis=0)
        is_initial = T.shape_padright(is_initial).repeat(n_strings, axis=1)
        initial_val = T.cast(i, theano.config.floatX)
        new_vals = T.switch(
            is_initial,
            initial_val,
            new_vals
        )

        return new_vals

    results, _ = theano.scan(
        step,
        sequences=[T.arange(i_max + 1)],
        outputs_info=dict(initial=T.zeros((2,j_max + 1,n_strings)), taps=[-1,-2])
    )

    return results[-1, -1, :]#.T

# # NOTE: If this ever becomes a bottleneck, I think there's a more
# # GPU-friendly implementation possible by scanning along diagonals instead
# # of rows/cols.
# def _levenshtein_distance(a, b):
#     """Calculate the Levenshtein distance between two sets of strings."""

#     # return T.arange(b.shape[0])+1
#     a = T.as_tensor_variable(a)
#     b = T.as_tensor_variable(b)

#     def scan_a(current_a, current_a_index, last_row):

#         def scan_b(current_b, current_b_index, last_score):
#             # return last_row[current_b_index] + 1
#             # return T.cast(T.eq(current_a, current_b), dtype='float32')
#             result = T.switch(
#                 T.eq(current_a, current_b),# * T.neq(current_a, current_b),
#                 last_row[current_b_index-1],
#                 # T.cast(T.eq(current_a, current_b), 'float32'),
#                 T.min([
#                     last_row[current_b_index] + 2,
#                     # last_score + 1,
#                     # last_row[current_b_index - 1] + 1
#                 ])
#             )
#             # return T.cast(T.eq(current_a, current_b), 'float32')
#             return result
#             return T.switch(
#                 T.eq(current_b, numpy.float32(-1)),
#                 last_score,
#                 result
#             )

#         results, _ = theano.scan(
#             scan_b,
#             sequences=[b, T.arange(b.shape[0])+1],
#             outputs_info=T.alloc(T.cast(current_a_index, theano.config.floatX), b.shape[1]) #T.arange(b.shape[1], dtype=theano.config.floatX)
#         )

#         # return T.switch(
#         #     T.eq(current_a, numpy.float32(-1)),
#         #     last_row,
#         return T.concatenate([T.alloc(T.cast(current_a_index, theano.config.floatX), 1, b.shape[1]), results])
#         # )

#     results, _ = theano.scan(
#         scan_a,
#         sequences=[a, T.arange(a.shape[0])+1],
#         outputs_info=T.arange(b.shape[0] + 1, dtype=theano.config.floatX).repeat(b.shape[1]).reshape((b.shape[0]+1, b.shape[1]))
#     )

#     return results[:,:,1]