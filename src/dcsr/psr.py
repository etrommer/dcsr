from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt


@dataclass
class PSRExport:
    values: npt.NDArray
    indices: npt.NDArray[np.uint8]
    flat_counts: npt.NDArray
    row_elements: List[int]

    @property
    def size(self) -> int:
        return self.values.nbytes + self.indices.nbytes + self.flat_counts.nbytes + len(self.row_elements) * 4


@dataclass
class PSRRow:
    values: npt.NDArray
    indices: npt.NDArray[np.uint8]
    counts: npt.NDArray[np.uint8]


class PSRMatrix:
    def __init__(self, matrix: npt.NDArray, index_width: int = 8):
        self.index_width = index_width
        self.rows = self.compress(matrix)

    def compress(self, matrix: npt.NDArray) -> List[PSRRow]:
        def compress_row(row: npt.NDArray) -> PSRRow:
            idxs = np.where(row != 0)[0]
            psr_idxs = (idxs % 2**self.index_width).astype(np.uint8)
            _, psr_counts = np.unique(idxs // 2**self.index_width, return_counts=True)
            psr_counts = psr_counts.astype(np.uint8)
            return PSRRow(row[idxs], psr_idxs, psr_counts)

        return [compress_row(r) for r in matrix]

    def export(self) -> PSRExport:
        return PSRExport(
            np.concatenate([r.values for r in self.rows]),
            np.concatenate([r.indices for r in self.rows]),
            np.concatenate([r.counts for r in self.rows]),
            [len(r.values) for r in self.rows],
        )

    @property
    def size(self) -> int:
        return self.export().size
