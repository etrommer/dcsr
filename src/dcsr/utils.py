import numpy as np
import numpy.typing as npt


# Pack Matrix of 4-bit delta groups to 8-bit values by storing even rows
# in upper nibble and odd rows in lower nibble
def pack_nibbles(array: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    assert np.all(array <= 0xF)
    upper = np.left_shift(array[::2].astype(np.int32), 4)
    lower = array[1::2]
    if lower.shape != upper.shape:
        lower = np.append(lower, np.zeros((1, lower.shape[1])), axis=0)
    return (upper + lower).astype(np.uint8)


def checked_conversion(matrix, dtype):
    try:
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        if min_val < np.iinfo(dtype).min or max_val > np.iinfo(dtype).max:
            raise TypeError("Overflow in Conversion - min: {} - max {}".format(min_val, max_val))
        return matrix.astype(dtype)
    except ValueError:
        return matrix.astype(dtype)
