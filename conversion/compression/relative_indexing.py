import numpy as np

from .dcsr import remove_zeros, checked_conversion, get_row_offsets

def compress_matrix_relative(matrix):
    indices = []
    values = []
    row_lens = []

    # Row-wise compression to relative indexing
    for row in matrix:
        i, v = compress_row(row)
        row_lens.append(len(v))
        indices.extend(i)
        values.extend(v)

    # We pack two four-bit indices into a byte
    # so we make sure there's an even number of indices
    if len(indices) % 2 == 1:
        indices.append(0)

    # Check that indices don't exceed a nibble
    indices = np.array(indices).astype(np.int)
    assert np.all(indices < 16)

    # Interleaving
    upper = indices[::2]
    lower = indices[1::2]
    indices = np.left_shift(upper, 4) + lower

    row_lens = np.array(row_lens)

    ans = {
        "values": np.array(values).astype(np.int8),
        "delta_indices": checked_conversion(indices, np.uint8),
        "row_offsets" : checked_conversion(row_lens, np.int16),
        "nnze": np.sum(row_lens),
    }
    return ans

# Convert a single matrix row to Relative Indexing
def compress_row(row):
    (indices, values) = remove_zeros(row)
    last_idx = 0

    relative_indices = []
    relative_values = []

    for i, v in zip(indices, values):
        while i - last_idx > 15:
            last_idx += 15
            relative_indices.append(15)
            relative_values.append(0)
        relative_indices.append(i - last_idx)
        relative_values.append(v)
        last_idx = i
    return relative_indices, relative_values
