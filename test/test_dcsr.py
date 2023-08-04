import numpy as np
from dcsr.compress import DCSRMatrix
from dcsr.decompress import decompress
from dcsr.utils import pack_nibbles


def test_pack_nibbles():
    array = np.array([0xA, 0xB, 0xC, 0xD, 0xE], dtype=np.uint8)
    assert np.array_equal(pack_nibbles(array), np.array([0xAB, 0xCD, 0xE0], dtype=np.uint8))

    array = np.array([[0xA, 0xB, 0xC, 0xD], [0x1, 0x2, 0x3, 0x4], [0x5, 0x6, 0x7, 0x8]], dtype=np.uint8)
    assert np.array_equal(
        pack_nibbles(array), np.array([[0xA1, 0xB2, 0xC3, 0xD4], [0x50, 0x60, 0x70, 0x80]], dtype=np.uint8)
    )


def test_single_group():
    matrix = np.zeros((1, 32), dtype=np.int8)
    matrix[0, ::2] = 1
    compressed = DCSRMatrix(matrix)
    result = decompress(compressed.export(), matrix.shape)
    assert np.allclose(matrix, result)


def test_two_groups():
    matrix = np.zeros((1, 64), dtype=np.int8)
    matrix[0, ::2] = 1
    compressed = DCSRMatrix(matrix)
    result = decompress(compressed.export(), matrix.shape)
    assert np.allclose(matrix, result)


def test_minimum():
    matrix = np.zeros((1, 64))
    matrix[0, :32:2] = 1
    matrix[0, 33::2] = 1
    compressed = DCSRMatrix(matrix)
    result = decompress(compressed.export(), matrix.shape)
    assert np.allclose(matrix, result)


def test_deltas():
    matrix = np.zeros((1, 64))
    matrix[0, :48:3] = 1
    compressed = DCSRMatrix(matrix)
    result = decompress(compressed.export(), matrix.shape)
    assert np.allclose(matrix, result)


def test_bitmaps():
    matrix = np.zeros((1, 128))
    matrix[0, :16] = 1
    compressed = DCSRMatrix(matrix)
    result = decompress(compressed.export(), matrix.shape)
    assert np.allclose(matrix, result)


def test_rows():
    matrix = np.zeros((2, 32), dtype=np.int8)
    matrix[:, ::2] = 1
    compressed = DCSRMatrix(matrix)
    result = decompress(compressed.export(), matrix.shape)
    assert np.allclose(matrix, result)


def test_short_groups():
    matrix = np.zeros((2, 32), dtype=np.int8)
    matrix[:, 0] = 1
    compressed = DCSRMatrix(matrix)
    result = decompress(compressed.export(), matrix.shape)
    assert np.allclose(matrix, result)


def test_random():
    sparsity = 0.7

    matrix = np.random.randint(-128, 127, (1024, 1024), dtype=np.int8)
    mask = np.random.random(matrix.shape)
    matrix[mask < sparsity] = 0

    compressed = DCSRMatrix(matrix)
    result = decompress(compressed.export(), matrix.shape)
    assert np.allclose(matrix, result)
