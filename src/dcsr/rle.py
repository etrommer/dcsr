from typing import Any, Dict

import numpy as np
import numpy.typing as npt
from dcsr.compress import checked_conversion


def rle(matrix: npt.NDArray) -> Dict[str, Any]:
    indices = []
    values = []
    row_lens = []

    # Convert a single matrix row to Relative Indexing
    def compress_row_rle(row):
        (indices, values) = (np.flatnonzero(row != 0), row[row != 0])
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

    # Row-wise compression to relative indexing
    for row in matrix:
        i, v = compress_row_rle(row)
        row_lens.append(len(v))
        indices.extend(i)
        values.extend(v)

    # We pack two four-bit indices into a byte
    # so we make sure there's an even number of indices
    if len(indices) % 2 == 1:
        indices.append(0)

    # Check that indices don't exceed a nibble
    indices_np: npt.NDArray = np.array(indices).astype(np.uint8)
    assert np.all(indices_np < 16)

    # Interleaving
    upper = indices_np[::2]
    lower = indices_np[1::2]
    indices_np = np.left_shift(upper, 4) + lower

    row_lens_np = np.array(row_lens)

    ans = {
        "values": np.array(values).astype(np.int8),
        "delta_indices": checked_conversion(indices_np, np.uint8),
        "row_offsets": checked_conversion(row_lens_np, np.int16),
        "nnze": np.sum(row_lens),
    }
    return ans
