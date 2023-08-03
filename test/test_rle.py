import numpy as np
from dcsr.rle import RLEMatrix


def test_empty():
    matrix = np.zeros((1, 1), dtype=np.uint8)
    rle = RLEMatrix(matrix)

    assert rle.padding == 0
    assert rle.size == matrix.shape[0] * 4


def test_no_padding():
    matrix = np.zeros((256, 15), dtype=np.uint8)
    matrix[:, -1] = 1
    rle = RLEMatrix(matrix)

    BASE_ELEMS_SIZE = np.count_nonzero(matrix) * matrix.itemsize
    assert rle.size == BASE_ELEMS_SIZE + BASE_ELEMS_SIZE / 2 + matrix.shape[0] * 4
    assert rle.padding == 0


def test_padding():
    matrix = np.zeros((256, 16), dtype=np.uint8)
    matrix[:, -1] = 1
    rle = RLEMatrix(matrix)

    BASE_ELEMS_SIZE = np.count_nonzero(matrix) * matrix.itemsize

    # One padding element for each payload element
    PADDING_ELEMS_SIZE = BASE_ELEMS_SIZE
    TOTAL_ELEMS_SIZE = BASE_ELEMS_SIZE + PADDING_ELEMS_SIZE

    # padding property returns a _count_, not a memory consumption
    assert rle.padding == PADDING_ELEMS_SIZE / matrix.itemsize
    assert rle.size == TOTAL_ELEMS_SIZE + TOTAL_ELEMS_SIZE / 2 + matrix.shape[0] * 4
