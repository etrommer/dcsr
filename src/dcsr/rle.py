from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt
from dcsr.compress import pack_nibbles


@dataclass
class RLEExport:
    values: npt.NDArray
    indices: npt.NDArray[np.uint8]
    row_lengths: List[int]

    @property
    def size(self) -> int:
        return self.values.nbytes + self.indices.nbytes + len(self.row_lengths) * 4


@dataclass
class RLERow:
    values: npt.NDArray
    indices: npt.NDArray[np.uint8]


class RLEMatrix:
    def __init__(self, matrix: npt.NDArray, index_width: int = 4):
        self.index_width = index_width
        self.rows = self.compress(matrix)

    def export(self) -> RLEExport:
        values = np.concatenate([r.values for r in self.rows])
        indices = np.concatenate([r.indices for r in self.rows])
        if self.index_width <= 4:
            indices = pack_nibbles(indices)
        row_lengths = [len(r.values) for r in self.rows]
        return RLEExport(values, indices, row_lengths)

    @property
    def padding(self) -> int:
        return sum([np.sum(r.values == 0) for r in self.rows])

    @property
    def size(self) -> int:
        return self.export().size

    @staticmethod
    def compress(matrix: npt.NDArray) -> List[RLERow]:
        # Convert a single matrix row to Relative Indexing
        def compress_row(row) -> RLERow:
            idxs = np.where(row != 0)[0]
            last_idx = 0
            vals = []
            rel_idxs = []
            for idx, v in zip(idxs, row[idxs]):
                difference = idx - last_idx
                overflows = difference // 15
                vals.extend([0] * overflows)
                rel_idxs.extend([15] * overflows)
                rel_idxs.append(difference % 15)
                vals.append(v)
                last_idx = idx
            return RLERow(np.array(vals, dtype=row.dtype), np.array(rel_idxs, dtype=np.uint8))

        return [compress_row(r) for r in matrix]
