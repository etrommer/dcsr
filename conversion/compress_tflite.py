#!/usr/bin/env python3
import flatbuffers
import tflite
import argparse
import json
import logging
import os

import numpy as np

from tflite.BuiltinOperator import BuiltinOperator as OpType
from tflite.TensorType import TensorType
from tflite.Model import ModelT

from compression.dcsr import compress_matrix
from compression.relative_indexing import compress_matrix_relative

'''
Helper script that parses a Tensorflow lite model, generates a list of candidate tensors and compresses them to deltaCSR.
Results can either be saved as a Tflite model or directly converted to a C array
'''

#Load TFlite file
def load_model(fh):
    buf = bytearray(fh.read())
    model = tflite.Model.Model.GetRootAsModel(buf, 0)
    return tflite.Model.ModelT.InitFromObj(model)

# Rebuild TFlite model
def build_model(model):
    b = flatbuffers.Builder(1024)
    b.Finish(model.Pack(b))
    return b.Output()

# Save to TFlite file
def save_model(model, fh):
    fh.write(model)

# Export weight matrix as numpy array
def save_tensor_weights(tensor_weights, base_path, name):
    path = os.path.join(base_path, name + '.npy')
    with open(path, 'wb') as f:
        np.save(f, tensor_weights)

# Convert Model to C array for compiling into TFlite micro
# This is a builtin replacement for xxd so that the additional
# conversion step from .tflite to C source file can be omitted
def save_c_array(model, fh):
    file_contents = "#include \"tensorflow/lite/micro/examples/hello_world/model.h\"\r\n\r\n"
    file_contents += "alignas(8) const unsigned char g_model[] = {\r\n"
    line_length = 12
    num_lines = int(np.floor(len(model)/line_length))
    for line in range(num_lines):
        line_content = "  " + ", ".join(["0x{:02x}".format(b) for b in model[line*line_length:(line+1)*line_length]])
        line_content += ",\r\n"
        file_contents += line_content
    file_contents += "  " + ", ".join(["0x{:02x}".format(b) for b in model[num_lines*line_length:]]) + "\r\n};\r\n"
    file_contents += "const int g_model_len = {};".format(len(model))

    fh.write(file_contents)

# Calculate the size of a single tensor
def tensor_size(model, subgraph, tensor):
    data = model.buffers[subgraph.tensors[tensor].buffer].data
    if data is None:
        return 0
    return len(data)

# Iterate nodes to get the size of weights and biases in the model
def model_size(model):
    weight_sizes = []
    bias_sizes = []
    opcodes = []

    s = model.subgraphs[0]
    for op in s.operators:
        if len(op.inputs) > 3:
            raise ValueError("Unsupported node type")

        # node with weights only
        if len(op.inputs) == 2:
            ws = tensor_size(model, s, op.inputs[1])
            weight_sizes.append(ws)
            bias_sizes.append(0)
        # node with weights and biases
        elif len(op.inputs) == 3:
            ws = tensor_size(model, s, op.inputs[1])
            bs = tensor_size(model, s, op.inputs[2])
            weight_sizes.append(ws)
            bias_sizes.append(bs)
        # operation node without parameters
        else:
            weight_sizes.append(0)
            bias_sizes.append(0)
        opcodes.append(op.opcodeIndex)
    return np.array(weight_sizes), np.array(bias_sizes), np.array(opcodes)

# Find weight tensors with format supported by packing
def compressible_tensors(model):
    weight_tensors = []
    s = model.subgraphs[0]
    graph_ops = model.operatorCodes

    for op in s.operators:
        op_type = graph_ops[op.opcodeIndex]
        # FC layers
        if op_type.deprecatedBuiltinCode == OpType.FULLY_CONNECTED:
            t = op.inputs[1]
            major_dim = np.prod(s.tensors[t].shape[:-1])
            minor_dim = s.tensors[t].shape[-1]
            weight_tensors.append(op.inputs[1])

        # Conv layers
        elif op_type.deprecatedBuiltinCode == OpType.CONV_2D:
            t = op.inputs[1]
            major_dim = np.prod(s.tensors[t].shape[:-1])
            minor_dim = s.tensors[t].shape[-1]

            # Ignore Conv layers that are not pointwise
            dims = s.tensors[t].shape
            if dims[1] == 1 and dims[2] == 1:
                weight_tensors.append(t)
    return weight_tensors

# Convert single tensor to deltaCSR
def compress_weight_matrix(weight_matrix):
    result, metrics = compress_matrix(weight_matrix)

    sparsity = tflite.SparsityParameters.SparsityParametersT()
    compressed_sparsity = tflite.CompressedSparsity.CompressedSparsityT()

    compressed_sparsity.deltaIndices = result['delta_indices']
    compressed_sparsity.groupMinimums = result['minimums']
    compressed_sparsity.bitmaps = result['bitmaps']
    compressed_sparsity.bitmasks = result['bitmasks']
    compressed_sparsity.rowOffsets = result['row_offsets']
    compressed_sparsity.nnze = result['nnze']

    sparsity.compSparsity = compressed_sparsity
    return sparsity, result['values'], metrics

# Convert single tensor to relative indexing
def compress_relative(weight_matrix):
    result = compress_matrix_relative(weight_matrix)

    sparsity = tflite.SparsityParameters.SparsityParametersT()
    compressed_sparsity = tflite.CompressedSparsity.CompressedSparsityT()

    compressed_sparsity.deltaIndices = result['delta_indices']
    compressed_sparsity.rowOffsets = result['row_offsets']
    compressed_sparsity.bitmaps = np.array([0xde, 0xad, 0xde, 0xad]).astype(np.uint8)

    compressed_sparsity.nnze = result['nnze']
    sparsity.compSparsity = compressed_sparsity
    return sparsity, result['values']

# Takes a list of sparse weight tensors and replaces their contents
# with their compressed representation
def compress_tensor_list(model, tensor_list, sparsity_threshold=0.7, save_weights=False, return_numpy=False, relative_indexing=False):
    s = model.subgraphs[0]
    ans = []
    for t in tensor_list:
        # Only the last dimension gets compressed. For reshaping, we simply squash all
        # higher dimensions into one by multiplying them together to obtain a 2D matrix
        major_dim = np.prod(s.tensors[t].shape[:-1])
        minor_dim = s.tensors[t].shape[-1]
        tensor_weights = model.buffers[s.tensors[t].buffer].data.reshape(major_dim, minor_dim).astype(np.int8)
        sparsity = np.sum(tensor_weights == 0)/(minor_dim * major_dim)

        tensor_name = s.tensors[t].name.decode()

        # Check if tensor has noticeable sparsity
        if sparsity < sparsity_threshold:
            logging.debug("Tensor {} with sparsity {} is below threshold {} - Skipping".format(
                tensor_name,
                sparsity,
                sparsity_threshold))
            continue

        # Extract weights as numpy array
        if return_numpy is True:
            ans.append(tensor_weights)
            continue

        # Extract weights as separate files
        # Useful for generating test files
        if save_weights is True:
            logging.debug("Saving Tensor {} as .npy".format(tensor_name))
            save_tensor_weights(tensor_weights, '../test_data', tensor_name.split('/')[1])
            continue

        if relative_indexing is True:
            # Get Relative Indexing
            logging.debug("Compressing {} to Relative Indexing".format(tensor_name))
            sparsity_info, values = compress_relative(tensor_weights)
        else:
            # Get dCSR
            sparsity_info, values, metrics = compress_weight_matrix(tensor_weights)
            # Print a few results
            logging.debug("Compressed Tensor {}, Dims: {}*{}, Sparsity {:.2f}".format(
                tensor_name,
                major_dim,
                minor_dim,
                sparsity))
            logging.debug("Size before: {:.2f}KiB, Size after: {:.2f}KiB, Compression Ratio: {}".format(
                metrics['size_dense']/2**10,
                metrics['size_sparse_dCSR']/2**10,
                metrics['size_dense']/metrics['size_sparse_dCSR']))

        # TODO: We assume 8-Bit quantization here. Consider making this a runtime parameter
        model.buffers[s.tensors[t].buffer].data = values.astype(np.int8)
        s.tensors[t].sparsity = sparsity_info

    return ans

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Repacking of Tensorflow lite models with sparse weight tensors"
    )
    parser.add_argument(
        'input', type=argparse.FileType('rb'),
        help="Tensorflow lite file to convert"
    )
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('wb'),
        help="Tensorflow lite file to write conversion result to"
    )
    parser.add_argument(
        '--cppmodel', '-c', type=argparse.FileType('w+'),
        help="C++ Source file to write array representation of result to"
    )
    parser.add_argument(
        '--compress', action='store_true',
        help="Apply deltaCSR packing to model"
    )
    parser.add_argument(
        '--relative', action='store_true',
        help="Apply Relative Indexing instead of deltaCSR"
    )
    parser.add_argument(
        '-d', '--debug', action='store_const', dest='loglevel', const=logging.DEBUG, default=logging.WARNING,
        help='Print Debug Output',
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)

    m = load_model(args.input)
    weight_sizes, bias_sizes, _ = model_size(m)
    logging.debug("Model size before compression - Weights: {:.2f} KiB, Biases: {:.2f} KiB".format(np.sum(weight_sizes)/2**10, np.sum(bias_sizes)/2**10))

    # deltaCSR compression can be skipped in order to create a dense reference model.
    # This is then simply a replacement for xxd that removes a few manual steps.
    if args.compress == True:
        compressible_tensors = compressible_tensors(m)
        compress_tensor_list(m, compressible_tensors, relative_indexing=args.relative)

    finalized_model = build_model(m)
    if args.cppmodel is not None:
        save_c_array(finalized_model, args.cppmodel)
    if args.output is not None:
        save_model(finalized_model, args.output)
