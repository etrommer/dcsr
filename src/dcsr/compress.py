from collections.abc import Iterable
from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt

import dcsr.utils
from dcsr import metrics
from dcsr.export import DCSRExport


@dataclass
class DCSRRow:
    values: npt.NDArray
    delta_indices: npt.NDArray[np.int16]
    minimums: npt.NDArray[np.int8]
    slope: int
    bitwidths: List[int]
    groups_count: int
    num_elements: int


class DCSRMatrix:
    def __init__(self, matrix: npt.NDArray, group_size: int = 16) -> None:
        if len(matrix.shape) != 2:
            raise ValueError("Only 2D matrices are supported.")
        # Compress individual rows
        self.base_matrix = matrix
        self.row_data: List[DCSRRow] = [self.compress_row(row) for row in matrix]

    def export(self) -> DCSRExport:
        """
        - Split Delta indices into
            - 4-Bit Base Index
            - 1 Extension Bitmap per group that marks which Bits need to be extended
            - A varying number of extension masks, depending on the number of bits set in the bitmap
        - Flatten everything to 1D arrays

        Returns:
            Dataclass with 1D arrays for each dCSR component
        """
        split_indices = []
        for r in self.row_data:
            # Get bitmaps and bitmasks for delta indices with >4-Bit
            (bitmaps, bitmasks) = self.get_merged_bitmaps(r.delta_indices, [5, 6, 7, 8])
            # Remove >4-Bit component from base_indices
            base_indices = np.bitwise_and(r.delta_indices, 0xF)
            split_indices.append((base_indices, bitmaps, bitmasks))

        base_indices = np.concatenate([r[0] for r in split_indices])
        base_indices = dcsr.utils.checked_conversion(dcsr.utils.pack_nibbles(base_indices).flatten(), np.uint8)

        # Some reshaping wizardry so that the pack_nibbles() function can be used for the bitmaps as well...
        bitmaps = np.concatenate([r[1] for r in split_indices])
        bitmaps = dcsr.utils.checked_conversion(
            dcsr.utils.pack_nibbles(bitmaps.reshape(len(bitmaps), 1)).flatten(),
            np.uint8,
        )

        bitmasks = np.concatenate([r[2] for r in split_indices])
        bitmasks = dcsr.utils.checked_conversion(self.convert_mask(bitmasks), np.uint16)

        export = DCSRExport(
            np.concatenate([r.values for r in self.row_data]).astype(self.base_matrix.dtype),
            base_indices,
            np.concatenate([r.minimums for r in self.row_data]).astype(np.int8),
            bitmaps,
            bitmasks,
            self.get_row_offsets([r.num_elements for r in self.row_data]).astype(np.int16),
            [r.slope for r in self.row_data],
            [r.num_elements for r in self.row_data],
            sum(r.num_elements for r in self.row_data),
        )
        return export

    @property
    def metrics(self) -> metrics.SparseMetrics:
        # Merge Results
        return metrics.get_metrics(self.export(), self.base_matrix)

    # Turn a sparse row into a list of column indices
    # and a list of values
    @staticmethod
    def remove_zeros(row):
        return (np.flatnonzero(row != 0), row[row != 0])

    # Add zeroes so that values array is a multiple of the group size
    @staticmethod
    def pad_values(values, group_size=16):
        if len(values) % group_size == 0:
            return values
        fill_elems = group_size - (len(values) % group_size)
        padded = np.append(values, np.zeros(fill_elems))
        return padded

    # Group column indices together so that they fit one run of SIMD lanes
    @staticmethod
    def simd_groups(col_idx, group_size=16):
        # Fill partially empty groups with zeroes
        if len(col_idx) % group_size != 0:
            len_padding = group_size - (len(col_idx) % group_size)
            col_idx = np.append(col_idx, np.zeros(len_padding))
        return col_idx.reshape(-1, group_size)

    # Check if index offsets within one group will overflow
    # Scatter/Gather instructions require offset values to be uint8
    @staticmethod
    def idx_overflow(delta_indices, slope):
        slope = np.linspace(0, slope * (delta_indices.shape[1] - 1), delta_indices.shape[1])
        return np.any((delta_indices + slope) >= 256)

    @staticmethod
    def pad_largest_gap(indices, nzes, row_len):
        # Find largest gap
        if len(indices) == 1:
            largest_gap = indices[0]
            indices = np.insert(indices, 0, np.floor(indices[0] / 2))
            nzes = np.insert(nzes, 0, 0)
            return indices, nzes

        shifted = np.roll(np.append(indices, row_len), 1)
        shifted[0] = 0
        gap_sizes = np.append(indices, row_len) - shifted
        largest_gap_idx = np.argmax(gap_sizes)
        largest_gap = np.max(gap_sizes)

        # Add zero in largest gap
        if largest_gap_idx < len(indices):
            padding_idx = indices[largest_gap_idx] - np.ceil(largest_gap / 2)
            indices = np.insert(indices, largest_gap_idx, padding_idx)
            nzes = np.insert(nzes, largest_gap_idx, 0)
        else:
            padding_idx = row_len - np.ceil(largest_gap / 2)
            indices = np.append(indices, padding_idx)
            nzes = np.append(nzes, 0)

        return indices, nzes

    # Calculate Integer slope of linear approximation function
    @staticmethod
    def get_slope(row_len, nnze):
        if nnze == 0:
            return 0
        # Effectively just a round-to-nearest. Could be done using round(row_len/nnze),
        # however it is implemented here using integer-only maths to avoid
        # possible inconsistencies with the runtime calculation
        return int((row_len + int(nnze / 2)) / nnze)

    # Calculate the deviation from the base slope for each element
    @staticmethod
    def remove_slope(groups, slope):
        base = np.linspace(0, slope * (groups.shape[1] - 1), groups.shape[1])
        return groups - base

    # Remove common offset for each SIMD group
    # return:
    # - Groups with removed offsets
    # - Array of the offsets
    @staticmethod
    def remove_minimums(groups):
        mins = np.min(groups, axis=1)
        return (groups - mins.reshape(-1, 1), mins.astype(np.int32))

    # Turn group minimums into base pointer steps
    # if absolute then the base pointer offset needs to be calculated independently for each group
    # if relative then the base pointer is calcultated as an offset from the last base pointer
    # From each step, the expected step width (slope * group_size) is removed to obtain smaller values
    @staticmethod
    def minimum_offsets(minimums, slope, group_size=16, absolute=False):
        if absolute:
            offset = np.linspace(0, (len(minimums) - 1) * slope * group_size, len(minimums)).astype(np.int32)
            minimums -= offset
        else:
            minimums[1:] -= minimums[:-1]
            minimums[1:] -= slope * group_size
        return minimums

    # Calculate minimum bit width required to encode each group
    @staticmethod
    def get_bit_widths(groups):
        return np.ceil(np.log2(np.max(groups, axis=1) + 1))

    # Get merged bitmaps and list of masks for groups.
    # Receives a list of bitwidths for extension
    # return:
    # -   Map for each group in which the nth bit corresponds to the nth element in the
    #     input bitwidth list. If the nth bit is set, this group needs to be bit extended
    #     for this bitwidth.
    # -   Bit masks for dynamic extension, sorted per group
    @staticmethod
    def get_merged_bitmaps(groups, bitwidths):
        bitmaps = np.zeros(groups.shape[0])
        masks = []
        for position, bitwidth in enumerate(bitwidths):
            mask = np.any(np.bitwise_and(groups, 2 ** (bitwidth - 1)), axis=1)
            bitmaps[mask] += 2**position

        for g in groups:
            for bitwidth in bitwidths:
                mask = np.bitwise_and(g, 2 ** (bitwidth - 1))
                if np.any(mask):
                    masks.append(mask != 0)
        return bitmaps, np.array(masks).reshape(-1, groups.shape[1])

    # Check if the nth bit is set for each group,
    # return:
    # - Binary mask over groups that is True if at least one element in the group has a set bit in the given position
    # - Binary mask per element for each group where at least one element has the nth bit set
    # - Input matrix with the given bit position zeroed out
    @staticmethod
    def get_bit_position(groups, position=5):
        mask = np.right_shift(groups, position - 1)
        mask = np.mod(mask, 2)
        mask = mask == 1
        group_mask = np.any(mask, axis=1)
        return (
            group_mask,
            mask[group_mask, :],
            np.where(mask, groups - 2 ** (position - 1), groups),
        )

    # Turn list of absolute offsets into stepwise differences
    @staticmethod
    def get_offset_differences(offsets):
        shifted = np.insert(offsets[:-1], 0, 0)
        return offsets - shifted

    # Convert Bitmap from Boolean array to Binary representation
    @staticmethod
    def convert_map(bitmap):
        exps = np.logspace(0, len(bitmap) - 1, len(bitmap), base=2)
        return np.sum(exps * bitmap)

    # Convert Bitmask from Boolean array to Binary representation
    @staticmethod
    def convert_mask(mask):
        exps = np.logspace(0, mask.shape[1] - 1, mask.shape[1], base=2)
        return np.sum(exps * mask, axis=1)

    @staticmethod
    def to_c_def(param, name):
        out = "#define "
        out += name.upper()
        if isinstance(param, Iterable):
            out += " {"
            out += ", ".join([str(d) for d in param.tolist()])
            out += "}"
        else:
            out += " " + str(param)
        out += "\r\n"
        return out

    # Index array is padded with zeroes intially. This creates a very high error for the
    # padded elements after removing the slope and offset. To remove the error, we set the padded
    # elements to the group minimum after removing the slope. This results in an error of zero
    # after removing the offset
    @staticmethod
    def compensate_padding(groups, active_elements):
        # Calculate how many elements in the last row are active
        active_elements %= groups.shape[1]
        if active_elements == 0:
            return groups
        min_value = np.min(groups[-1, :active_elements])
        groups[-1, active_elements:] = min_value
        return groups

    @staticmethod
    def get_row_offsets(nnze_per_row):
        avg_nnze_per_row = np.floor(np.mean(nnze_per_row))
        row_offsets = nnze_per_row - avg_nnze_per_row
        return row_offsets

    def compress_row(self, row, slope=None, group_size=16, pad_to_groupsize=True) -> DCSRRow:
        (indices, values) = self.remove_zeros(row)

        # Padding insertion to ensure base types don't overflow
        while True:
            slope = self.get_slope(len(row), len(values))
            delta_indices = self.simd_groups(indices, group_size=group_size)
            delta_indices = self.remove_slope(delta_indices, slope)
            delta_indices = self.compensate_padding(delta_indices, len(indices))
            (delta_indices, ms) = self.remove_minimums(delta_indices)
            delta_indices = delta_indices.astype(np.uint8)
            mins = self.minimum_offsets(ms, slope, group_size=group_size)
            bitwidths = self.get_bit_widths(delta_indices)

            overflow = self.idx_overflow(delta_indices, slope)

            # Check that there are no potential overflows
            if (len(values) == 0) or (
                (not overflow)
                and (np.min(mins) > np.iinfo(np.int8).min)
                and (np.max(mins) < np.iinfo(np.int8).max)
                and (np.max(bitwidths) < 9)
            ):
                break
            (indices, values) = self.pad_largest_gap(indices, values, len(row))

        self.delta_indices = delta_indices

        row_data = DCSRRow(
            values,
            delta_indices,
            mins,
            slope,
            bitwidths,
            len(delta_indices),
            len(values),
        )
        return row_data


if __name__ == "__main__":
    # Minimal working example
    np.random.seed(23)
    sparsity = 0.8
    sparse_mat = np.random.randint(-128, 127, (64, 64), dtype=np.int8)
    mask = np.random.random(sparse_mat.shape)
    sparse_mat[mask < sparsity] = 0
    compressed_matrix = DCSRMatrix(sparse_mat)
    print(compressed_matrix.metrics)
