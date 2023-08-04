"""
Helper script that converts a .npy array into a C source file.
The source file will contain the deltaCSR representation of the input matrix in the form of several arrays.
The generated source can be used with the testing implementation to verify the correctness of results 
for a specific array
"""

import argparse

import numpy as np
from dcsr.compress import DCSRMatrix


# Insert buffer array without data
def plain_array(name, base_type, size, description=None):
    ans = ""
    if description is not None:
        ans += "// {}\r\n".format(description)
    ans += "{} {}[{}];\r\n".format(base_type, name, size)
    return ans


# Numpy array to C array
def to_c_array(array, name, base_type, description=None):
    ans = ""
    if description is not None:
        ans += "// {}\r\n".format(description)
    ans += "const " + base_type + " " + name + "[] " + " = {\r\n"
    line_length = 12
    num_lines = int(np.floor(len(array) / line_length))
    for line in range(num_lines):
        ans += (
            "    "
            + ", ".join(["{:4d}".format(a) for a in array[line * line_length : (line + 1) * line_length]])
            + ",\r\n"
        )
    if len(array) % line_length == 0:
        ans += "\r\n};\r\n"
    else:
        ans += "    " + ", ".join(["{:4d}".format(a) for a in array[num_lines * line_length :]]) + "\r\n};\r\n"
    return ans


# Scalar value to C const
def to_c_const(value, name, base_type, description=None):
    ans = ""
    if description is not None:
        ans += "// {}\r\n".format(description)
    ans += "const " + base_type + " " + name + " = " + str(value) + ";\r\n"
    return ans


def cli():
    parser = argparse.ArgumentParser(description="Generate test file with C arrays from .npy array for testing")
    parser.add_argument("input", type=argparse.FileType("rb"), help="Numpy array in .npy format")
    parser.add_argument("output", type=argparse.FileType("w+"), help="C source file to write output to")

    args = parser.parse_args()

    matrix = np.load(args.input)
    if matrix.ndim != 2:
        raise ValueError("Can only compress 2D arrays. Please reshape array.")

    compressed = DCSRMatrix(matrix).export()

    dummy_input = np.random.randint(-128, 127, size=(4, matrix.shape[1]), dtype=np.int8)

    args.output.write('#include "test_matrix.h"\r\n\r\n')

    # Reference matrix without compression
    args.output.write(
        to_c_array(matrix.flatten().astype(np.int8), "reference", "int8_t", "Non-compressed array for comparision")
    )

    # dCSR data
    args.output.write(to_c_array(compressed.values, "values", "int8_t", "Compressed sparse values"))
    args.output.write(to_c_array(compressed.bitmaps, "bitmaps", "uint8_t", "Extension Bitmaps"))
    args.output.write(to_c_array(compressed.bitmasks, "bitmasks", "uint16_t", "Groupwise Bitmask"))
    args.output.write(to_c_array(compressed.delta_indices, "delta_indices", "uint8_t", "Delta Index 4-Bit Base Values"))
    args.output.write(to_c_array(compressed.minimums, "minimums", "int8_t", "Groupwise Base pointer steps"))
    args.output.write(
        to_c_array(
            compressed.row_offsets,
            "row_offsets",
            "int16_t",
            "Deviation of Row length from average row length",
        )
    )
    args.output.write(
        to_c_const(compressed.nnze, "nnze", "uint32_t", "Number of non-zero elements in the sparse matrix")
    )

    args.output.write(to_c_const(matrix.shape[0], "matrix_rows", "uint32_t", "Rows of dense matrix"))
    args.output.write(to_c_const(matrix.shape[1], "matrix_cols", "uint32_t", "Columns of dense matrix"))

    # Random input data to evaluate correctness of SpMM and SpMV results
    args.output.write(to_c_array(dummy_input.flatten(), "dummy_input", "int8_t", "Random dummy input"))

    # Pre-definded Row and group buffer arrays of correct size
    args.output.write(plain_array("idx_buffer", "uint8_t", matrix.shape[1], "Buffer for one row of indices"))
    args.output.write(
        plain_array(
            "group_buffer", "int16_t", int(np.ceil(matrix.shape[1] / 16)), "Buffer for one row of base pointer steps"
        )
    )


if __name__ == "__main__":
    cli()
