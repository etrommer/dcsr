import zlib

import numpy as np


def estimate_bcsr(matrix, block_size=2, row_ptr_bytes=2, col_idx_bytes=2, value_bytes=1):
    nz_blocks = 0
    for i in range(0, matrix.shape[0], block_size):
        for j in range(0, matrix.shape[1], block_size):
            block = matrix[i : i + block_size, j : j + block_size]
            if np.any(block != 0):
                nz_blocks += 1
    return nz_blocks * block_size**2 * value_bytes + nz_blocks * col_idx_bytes + matrix.shape[0] * row_ptr_bytes


def estimate_csr(matrix, row_ptr_bytes=2, col_idx_bytes=2, value_bytes=1):
    nzes = np.sum(matrix != 0)
    return nzes * value_bytes + nzes * col_idx_bytes + matrix.shape[0] * row_ptr_bytes


def estimate_relative_indexing(matrix, encoding_bits=4, row_ptr_bytes=2, value_bytes=1):
    total_size = 0
    padding_sum = 0
    for row in matrix:
        idxs = np.flatnonzero(row != 0)
        if len(idxs) == 0:
            continue
        # Calculate inter-element gaps
        shifted = np.roll(idxs, 1)
        shifted[0] = 0
        relative_idxs = idxs - shifted
        # Number of gaps that need padding elements
        padding_elems = np.sum(np.floor(relative_idxs / 2**encoding_bits))

        # Size of values array after padding + n bits of index per element + row_ptr
        total_size += (len(idxs) + padding_elems) * (value_bytes + encoding_bits / 8) + row_ptr_bytes
        padding_sum += padding_elems
    return total_size, padding_sum


def estimate_zlib(matrix, level=6):
    return len(zlib.compress(matrix.tobytes(), level=level))
