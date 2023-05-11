from typing import Any, Dict

import numpy as np
from more_itertools import chunked


def decompress(compressed_data: Dict[str, Any], shape) -> np.ndarray:
    buffer = np.zeros(shape, dtype=compressed_data["values"].dtype)
    GROUP_SIZE = 16

    # Decompress bitmaps from packed nibbles
    bitmaps = []
    for bmp in compressed_data["bitmaps"]:
        bitmaps.append((bmp & 0xF0) >> 4)
        bitmaps.append(bmp & 0xF)

    # Decompress delta values from 16 interleaved nibbles
    deltas = []
    for d in chunked(compressed_data["delta_indices"], GROUP_SIZE):
        deltas.append((np.array(d) & 0xF0) >> 4)
        deltas.append(np.array(d) & 0xF)

    # Tracking pointers
    mask_idx = 0
    value_start_idx = 0
    group_count = 0

    # Decompress each row
    for row, slope, num_elems in zip(buffer, compressed_data["slope"], compressed_data["num_elements"]):
        base_ptr = 0
        num_groups = int(np.ceil(num_elems / GROUP_SIZE))

        # Decompress each group
        for group_idx in range(num_groups):
            delta = deltas[group_count]
            bitmap = bitmaps[group_count]
            minimum = compressed_data["minimums"][group_count]

            active_elems = min(num_elems - (GROUP_SIZE * group_idx), GROUP_SIZE)
            extracted_idx: np.ndarray = np.arange(GROUP_SIZE) * slope + delta
            for shl in range(4):
                if (bitmap >> shl) & 1:
                    mask = compressed_data["bitmasks"][mask_idx]
                    mask_idx += 1
                    mask_array = np.array(
                        [(mask & (1 << bit_position)) >> bit_position for bit_position in range(GROUP_SIZE)]
                    )
                    extracted_idx += np.left_shift(mask_array, shl + 4)

            extracted_idx += base_ptr + minimum

            values = compressed_data["values"][value_start_idx : value_start_idx + active_elems]
            row[extracted_idx[:active_elems]] = values

            value_start_idx += active_elems
            base_ptr += minimum + GROUP_SIZE * slope
            group_count += 1
    return buffer
