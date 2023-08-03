from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt


@dataclass
class DCSRRow:
    values: npt.NDArray
    delta_indices: npt.NDArray[np.int16]
    minimums: npt.NDArray[np.int8]
    slope: int
    bitwidths: List[int]
    groups_count: int
    num_elements: int


@dataclass
class DCSRExport:
    values: npt.NDArray

    delta_indices: npt.NDArray[np.uint8]
    minimums: npt.NDArray[np.int8]
    bitmaps: npt.NDArray[np.uint8]
    bitmasks: npt.NDArray[np.uint16]

    row_offsets: npt.NDArray[np.int16]
    slope: List[int]
    num_row_elements: List[int]
    nnze: int
