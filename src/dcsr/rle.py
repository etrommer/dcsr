from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt

from dcsr.utils import pack_nibbles


@dataclass
class RLEExport:
    """
    Flat and typed data structures that can be written out to a C array etc.
    """

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
        """
        Compress matrix using Run-Length Encoding/Relative Indexing.

        Encodes each row as a series of relative indices, where each element's index
        describes the offset from the previous element. If the distance would overflow,
        a padding element is inserted.

        Args:
            matrix: A sparse matrix to compress
            index_width: Number of _bits_ to use for each element index. Defaults to 4.
        """
        assert matrix.ndim == 2, "Bad input dimensionality. Reshape input to 2D."
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
        """
        Number of inserted padding elements
        """
        return sum([np.sum(r.values == 0) for r in self.rows])

    @property
    def size(self) -> int:
        """
        Memory consumption in Bytes
        """
        return self.export().size

    def compress(self, matrix: npt.NDArray) -> List[RLERow]:
        def compress_row(row) -> RLERow:
            # Convert a single matrix row to Relative Indexing
            idxs = np.where(row != 0)[0]
            last_idx = 0
            vals = []
            rel_idxs = []
            max_elem = 2**self.index_width - 1
            for idx, v in zip(idxs, row[idxs]):
                difference = idx - last_idx
                overflows = difference // max_elem
                vals.extend([0] * overflows)
                rel_idxs.extend([max_elem] * overflows)
                rel_idxs.append(difference % max_elem)
                vals.append(v)
                last_idx = idx
            return RLERow(np.array(vals, dtype=row.dtype), np.array(rel_idxs, dtype=np.uint8))

        return [compress_row(r) for r in matrix]
