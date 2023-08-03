import numpy as np
from dcsr.psr import PSRMatrix


def test_empty():
    matrix = np.zeros((1, 1))
    psr = PSRMatrix(matrix)

    assert psr.size == matrix.shape[0] * 4


def test_single_submatrix():
    matrix = np.ones((1, 256))
    psr = PSRMatrix(matrix)

    BASE_ELEMS_COUNT = np.count_nonzero(matrix)
    INDEX_ELEMS_COUNT = BASE_ELEMS_COUNT
    GROUP_COUNT_ARRAYS = 1

    assert psr.size == BASE_ELEMS_COUNT * matrix.itemsize + INDEX_ELEMS_COUNT + GROUP_COUNT_ARRAYS + matrix.shape[0] * 4


def test_multiple_submatrix():
    matrix = np.ones((1, 1024))
    psr = PSRMatrix(matrix)

    BASE_ELEMS_COUNT = np.count_nonzero(matrix)
    INDEX_ELEMS_COUNT = BASE_ELEMS_COUNT
    GROUP_COUNT_ARRAYS = 4

    assert psr.size == BASE_ELEMS_COUNT * matrix.itemsize + INDEX_ELEMS_COUNT + GROUP_COUNT_ARRAYS + matrix.shape[0] * 4
