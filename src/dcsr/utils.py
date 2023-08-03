from typing import Any, List, Optional, Union

import numpy as np
import numpy.typing as npt


class CVariable:
    def __init__(self, data: Any, name: str, dtype: str, description: Optional[str] = None):
        """
        Turn a value into a string representing a C variable definition

        Args:
            data: Data item
            name: Name for the C variable
            dtype: data type for the variable definition
            description: Optional comment to add to the definition. Defaults to None.
        """
        self.data = data
        self.name = name
        self.dtype = dtype
        self.description = description

    def __repr__(self) -> str:
        ans = ""
        if self.description is not None:
            ans += f"//{self.description}\r\n"
        ans += f"const {self.dtype} {self.name} = {str(self.data)};\r\n"
        return ans


class CArray:
    def __init__(self, data: Union[List, npt.NDArray], name: str, dtype: Optional[str] = None):
        """
        Turn a Python list or numpy array into a string representing a C array definition

        Args:
            data: Underlying data
            name: Name for the C array
            dtype: data type for array definition. Can be inferred if `data` is a numpy array.
                Uses `int32_t` if not specified.
                Defaults to None.
        """
        if dtype is None:
            if isinstance(data, np.ndarray):
                self.dtype = str(data.dtype) + "_t"
            else:
                self.dtype = "int32_t"
        else:
            self.dtype = dtype
        self.data = data
        self.name = name
        self.line_length = 12

    def __repr__(self) -> str:
        ans = f"const {self.dtype} {self.name}[] BENCH_DATA = "
        if len(self.data) <= self.line_length:
            # Entire array fits on one line
            return ans + "\r\n    {" + ",".join([f"{i:4d}" for i in self.data]) + "};\r\n"
        else:
            ans += "{\r\n"
            num_lines = len(self.data) // self.line_length
            for line in range(num_lines):
                # Write full lines
                ans += (
                    "    "
                    + ",".join([f"{a:4d}" for a in self.data[line * self.line_length : (line + 1) * self.line_length]])
                    + ",\r\n"
                )
            # Handle leftovers
            ans += (
                "    " + ",".join([f"{a:4d}" for a in self.data[num_lines * self.line_length : len(self.data)]]) + "}"
            )
        ans += ";\r\n"
        return ans


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
