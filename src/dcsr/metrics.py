import zlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from dcsr.psr import PSRMatrix
from dcsr.rle import RLEMatrix
from dcsr.structs import DCSRExport
from scipy.stats import entropy


@dataclass
class DCSRMetrics:
    num_padding_elements: int
    num_groups: int
    num_bitmaps: int
    num_bitmasks: int
    bytes_values: int
    bytes_delta_index: int
    bytes_bitmaps: int
    bytes_bitmasks: int
    bytes_group_minimums: int
    bytes_row_offsets: int
    payload: int
    base_type_size: int

    @property
    def total_index(self):
        return (
            self.bytes_bitmaps
            + self.bytes_bitmasks
            + self.bytes_delta_index
            + self.bytes_group_minimums
            + self.bytes_row_offsets
            + np.dtype(np.uint32).itemsize
        )

    @property
    def index_overhead(self):
        return self.total_index / (self.nnze * self.base_type_size)


@dataclass
class SparseMetrics:
    nnze: int
    bytes_dense: int
    csr: int
    bcsr: int
    psr: int
    rle: int
    rle_padding: int
    zlib6: int
    zlib9: int
    dcsr: int
    dcsr_info: DCSRMetrics


# Calculate some metrics from compressed matrix
def get_metrics(export: DCSRExport, matrix: npt.NDArray) -> SparseMetrics:
    nnze = np.count_nonzero(matrix)

    # Gather detailed metrics for dCSR
    dcsr_info = DCSRMetrics(
        num_padding_elements=len(export.values) - nnze,
        num_groups=len(export.minimums),
        num_bitmaps=len(export.bitmaps),
        num_bitmasks=len(export.bitmasks),
        bytes_values=export.values.nbytes,
        bytes_delta_index=export.delta_indices.nbytes,
        bytes_bitmaps=export.bitmaps.nbytes,
        bytes_bitmasks=export.bitmasks.nbytes,
        bytes_group_minimums=export.minimums.nbytes,
        bytes_row_offsets=export.row_offsets.nbytes,
        payload=nnze,
        base_type_size=matrix.itemsize,
    )

    # Gather metrics for reference formats
    rle = RLEMatrix(matrix)
    psr = PSRMatrix(matrix)
    metrics = SparseMetrics(
        bytes_dense=matrix.nbytes,
        nnze=nnze,
        csr=csr(matrix),
        bcsr=bcsr(matrix),
        psr=psr.size,
        rle=rle.size,
        rle_padding=rle.padding,
        zlib6=zlib_compression(matrix, 6),
        zlib9=zlib_compression(matrix, 9),
        dcsr=dcsr_info.total_index + dcsr_info.bytes_values,
        dcsr_info=dcsr_info,
    )

    return metrics


def bcsr(
    matrix: npt.NDArray, block_size: int = 2, row_ptr_bytes: int = 2, col_idx_bytes: int = 2, value_bytes: int = 1
) -> int:
    nz_blocks = 0
    for i in range(0, matrix.shape[0], block_size):
        for j in range(0, matrix.shape[1], block_size):
            block = matrix[i : i + block_size, j : j + block_size]
            if np.any(block != 0):
                nz_blocks += 1
    return nz_blocks * block_size**2 * value_bytes + nz_blocks * col_idx_bytes + matrix.shape[0] * row_ptr_bytes


def csr(matrix: npt.NDArray, row_ptr_bytes: int = 2, col_idx_bytes: int = 2, value_bytes: int = 1) -> int:
    nzes = np.sum(matrix != 0)
    return nzes * value_bytes + nzes * col_idx_bytes + matrix.shape[0] * row_ptr_bytes


def zlib_compression(matrix: npt.NDArray, level: int = 6) -> int:
    return len(zlib.compress(matrix.tobytes(), level=level))


def entropy_compression(matrix: npt.NDArray) -> int:
    _, counts = np.unique(matrix.flatten(), return_counts=True)
    return entropy(counts)
