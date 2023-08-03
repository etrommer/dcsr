import numpy as np
import numpy.typing as npt


# Pack Matrix of 4-bit delta groups to 8-bit values by storing even rows
# in upper nibble and odd rows in lower nibble
def pack_nibbles(array: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """
    Compresses an array of subsequent values between 0 and 15 into an array of bytes where the upper nibble is occupied
    by the elements at even positions and the lower nibble is occupied by elements at odd positions in the original
    array.

    Length of the input array is padded to be divisible by two.

    Args:
        array: Input array of values in [0,15]

    Returns:
        Compressed array of length ceil(len(array)/2)
    """
    assert np.all(array <= 0xF)
    assert array.ndim <= 2, "Only 1D and 2D arrays are supported for packing"

    upper = np.left_shift(array[::2].astype(np.int32), 4)
    lower = array[1::2]
    if lower.shape != upper.shape:
        lower = np.insert(lower, lower.shape[0], 0, axis=0)
    return (upper + lower).astype(np.uint8)


def checked_conversion(matrix: npt.NDArray, dtype: np.dtype):
    """
    Convert a numpy array to a given data type with overflow checking

    Args:
        matrix: The numpy array to convert
        dtype: Target data type

    Raises:
        TypeError: Conversion would lead to overflow

    Returns:
        Array with new data type
    """
    try:
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        if min_val < np.iinfo(dtype).min or max_val > np.iinfo(dtype).max:
            raise TypeError("Overflow in Conversion - min: {} - max {}".format(min_val, max_val))
        return matrix.astype(dtype)
    except ValueError:
        return matrix.astype(dtype)
