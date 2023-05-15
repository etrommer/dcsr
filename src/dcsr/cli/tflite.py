#!/usr/bin/env python3
import argparse
import logging
import os
from typing import BinaryIO, List, Optional, TextIO, Tuple

import flatbuffers
import numpy as np
import numpy.typing as npt
from dcsr import tflite_schema
from dcsr.compress import DCSRMatrix
from dcsr.rle import rle

"""
Helper script that parses a Tensorflow lite model, generates a list of candidate tensors and 
compresses them to deltaCSR. Results can either be saved as a Tflite model or directly 
converted to a C array
"""


class TFLiteModel:
    def __init__(self, fh: BinaryIO, sparsity_threshold: float = 0.7) -> None:
        buf = bytearray(fh.read())
        model = tflite_schema.Model.GetRootAsModel(buf, 0)
        self.model = tflite_schema.ModelT.InitFromObj(model)
        self.sparsity_threshold = sparsity_threshold

    def store(self, fh_tflite: Optional[BinaryIO], fh_array: Optional[TextIO]) -> None:
        fb_builder = flatbuffers.Builder(1024)
        fb_builder.Finish(self.model.Pack(fb_builder))
        bin_model = fb_builder.Output()
        if fh_tflite:
            fh_tflite.write(bin_model)
        if fh_array:
            fh_array.write(self.to_csrc(bin_model))

    # Find weight tensors with format supported by packing
    @property
    def compressible_tensors(self) -> List[int]:
        # Check if tensor type and shape are supported and that tensor has suffienct sparsity
        def op_type_supported(op_type: tflite_schema.OperatorCodeT, shape: Tuple[int, ...]) -> bool:
            if op_type.builtinCode == tflite_schema.BuiltinOperator.FULLY_CONNECTED:
                return True
            # Convolutions with Kernel size = 1 only
            if op_type.builtinCode == tflite_schema.BuiltinOperator.CONV_2D and shape[1] == 1 and shape[2] == 1:
                return True
            return False

        subgraph = self.model.subgraphs[0]
        graph_ops = self.model.operatorCodes

        # Find compressible tensors
        compressible_tensors = []
        for op in subgraph.operators:
            op_type = graph_ops[op.opcodeIndex]

            try:
                op_weight_tensor = op.inputs[1]
                weights = self.get_tensor_array(op_weight_tensor)
            except (IndexError, ValueError):
                continue

            if not op_type_supported(op_type, weights.shape):
                continue

            sparsity = (weights.size - np.count_nonzero(weights)) / weights.size
            if sparsity < self.sparsity_threshold:
                tensor_name = self.get_tensor_name(op_weight_tensor)
                logging.debug(
                    "Tensor {} with sparsity {} is below threshold {} - Skipping".format(
                        tensor_name, sparsity, self.sparsity_threshold
                    )
                )
                continue
            compressible_tensors.append(op_weight_tensor)
        return compressible_tensors

    def get_tensor_name(self, tensor_idx: int) -> str:
        return self.model.subgraphs[0].tensors[tensor_idx].name.decode()

    def get_tensor_array(self, tensor_idx: int, reshape_2d: bool = False) -> npt.NDArray:
        subgraph = self.model.subgraphs[0]
        tensor_buff = subgraph.tensors[tensor_idx].buffer
        tensor_array = self.model.buffers[tensor_buff].data
        shape = subgraph.tensors[tensor_idx].shape
        if reshape_2d:
            major_dim = np.prod(shape[:-1])
            minor_dim = shape[-1]
            shape = (major_dim, minor_dim)
        tensor_array = tensor_array.reshape(shape)
        return tensor_array

    def export_numpy(self, base_path: str) -> None:
        for t in self.compressible_tensors:
            weights = self.get_tensor_array(t, reshape_2d=True)
            name = self.get_tensor_name(t)
            path = os.path.join(base_path, name.split("/")[1] + ".npy")
            logging.debug("Exporting {} to {}".format(name, path))
            with open(path, "wb") as f:
                np.save(f, weights)

    @staticmethod
    def report(weights: npt.NDArray, name: str, method: str, size_before: int, size_after: int) -> None:
        logging.debug(
            "Compressed Tensor {} using {} method, Dims: {}*{}, Sparsity {:.2f}".format(
                name,
                method,
                weights.shape[0],
                weights.shape[1],
                (weights.size - np.count_nonzero(weights)) / weights.size,
            )
        )
        logging.debug(
            "Size before: {:.2f}KiB, Size after: {:.2f}KiB, Compression Ratio: {}".format(
                size_before / 2**10,
                size_after / 2**10,
                size_before / size_after,
            )
        )

    # Convert single tensor to Run-Length Encoding (RLE)
    @staticmethod
    def compress_rle(weights: npt.NDArray, name: str) -> Tuple[tflite_schema.SparsityParametersT, npt.NDArray[np.int8]]:
        result = rle(weights)

        sparsity = tflite_schema.SparsityParametersT()
        compressed_sparsity = tflite_schema.CompressedSparsityT()

        compressed_sparsity.deltaIndices = result["delta_indices"]
        compressed_sparsity.rowOffsets = result["row_offsets"]
        compressed_sparsity.bitmaps = np.array([0xDE, 0xAD, 0xDE, 0xAD]).astype(np.uint8)

        compressed_sparsity.nnze = result["nnze"]
        sparsity.compSparsity = compressed_sparsity

        total_storage = result["delta_indices"].nbytes + result["row_offsets"].nbytes + result["values"].nbytes

        TFLiteModel.report(weights, name, "RLE", weights.nbytes, total_storage)
        return sparsity, result["values"]

    # Convert single tensor to deltaCSR
    @staticmethod
    def compress_dcsr(weights: npt.NDArray, name: str) -> Tuple[tflite_schema.SparsityParametersT, npt.NDArray]:
        # result, metrics = compress_matrix(weight_matrix)
        dcsr_matrix = DCSRMatrix(weights)
        export = dcsr_matrix.export()

        sparsity = tflite_schema.SparsityParametersT()
        compressed_sparsity = tflite_schema.CompressedSparsityT()

        compressed_sparsity.deltaIndices = export.delta_indices
        compressed_sparsity.groupMinimums = export.minimums
        compressed_sparsity.bitmaps = export.bitmaps
        compressed_sparsity.bitmasks = export.bitmasks
        compressed_sparsity.rowOffsets = export.row_offsets
        compressed_sparsity.nnze = export.nnze

        sparsity.compSparsity = compressed_sparsity

        metrics = dcsr_matrix.metrics
        TFLiteModel.report(weights, name, "dCSR", metrics.bytes_dense, metrics.dcsr)

        return sparsity, export.values

    def compress(self, method: str) -> None:
        subgraph = self.model.subgraphs[0]
        for t in self.compressible_tensors:
            name = self.get_tensor_name(t)
            weights = self.get_tensor_array(t, reshape_2d=True)

            if method == "rle":
                # Get Relative Indexing
                sparsity_info, values = self.compress_rle(weights, name)
            elif method == "dcsr":
                # Get dCSR
                sparsity_info, values = self.compress_dcsr(weights, name)
            else:
                return

            # TODO: We assume 8-Bit quantization here. Consider making this a runtime parameter
            self.model.buffers[subgraph.tensors[t].buffer].data = values.astype(np.int8)
            subgraph.tensors[t].sparsity = sparsity_info

    # Convert Model to C array for compiling into TFlite micro
    # This is a builtin replacement for xxd so that the additional
    # conversion step from .tflite to C source file can be omitted
    @staticmethod
    def to_csrc(bin_model) -> str:
        file_contents = '#include "tensorflow/lite/micro/examples/hello_world/model.h"\r\n\r\n'
        file_contents += "alignas(8) const unsigned char g_model[] = {\r\n"
        LINE_LENGTH = 12
        num_lines = int(np.floor(len(bin_model) / LINE_LENGTH))
        for line in range(num_lines):
            line_content = "  " + ", ".join(
                ["0x{:02x}".format(b) for b in bin_model[line * LINE_LENGTH : (line + 1) * LINE_LENGTH]]
            )
            line_content += ",\r\n"
            file_contents += line_content
        file_contents += (
            "  " + ", ".join(["0x{:02x}".format(b) for b in bin_model[num_lines * LINE_LENGTH :]]) + "\r\n};\r\n"
        )
        file_contents += "const int g_model_len = {};".format(len(bin_model))
        return file_contents

    # Iterate nodes to get the size of weights and biases in the model
    def sizes(self) -> npt.NDArray:
        out = []

        # Calculate the size of a single tensor
        def tensor_size(subgraph, tensor):
            data = self.model.buffers[subgraph.tensors[tensor].buffer].data
            return 0 if data is None else len(data)

        subgraph = self.model.subgraphs[0]
        for op in subgraph.operators:
            weight_size = 0
            bias_size = 0
            if len(op.inputs) > 3:
                raise ValueError("Unsupported node type")
            # node with weights only
            if len(op.inputs) == 2:
                weight_size = tensor_size(subgraph, op.inputs[1])
                bias_size = 0
            # node with weights and biases
            elif len(op.inputs) == 3:
                weight_size = tensor_size(subgraph, op.inputs[1])
                bias_size = tensor_size(subgraph, op.inputs[2])
            # operation node without parameters
            out.append((weight_size, bias_size, op.opcodeIndex))
        return np.array(out)


def cli():
    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)

    parser = argparse.ArgumentParser(description="Repacking of Tensorflow lite models with sparse weight tensors")
    parser.add_argument("input", type=argparse.FileType("rb"), help="Tensorflow lite file to convert")
    parser.add_argument(
        "--output",
        "-o",
        type=argparse.FileType("wb"),
        help="Tensorflow lite file to write conversion result to",
    )
    parser.add_argument(
        "--cppmodel",
        "-c",
        type=argparse.FileType("w+"),
        help="C++ Source file to write array representation of result to",
    )
    parser.add_argument(
        "-m" "--method",
        dest="method",
        choices=["dcsr", "rle", "none"],
        default=["dcsr"],
        help="Choice of compression method",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
        help="Print Debug Output",
    )

    parser.add_argument(
        "--numpy",
        type=dir_path,
        help="Export compressible tensors as .npy numpy arrays. This can be helpful to generate testing data.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)

    m = TFLiteModel(args.input)
    sizes = m.sizes()
    logging.debug(
        "Model size before compression - Weights: {:.2f} KiB, Biases: {:.2f} KiB".format(
            np.sum(sizes[:, 0]) / 2**10, np.sum(sizes[:, 1]) / 2**10
        )
    )

    # # deltaCSR compression can be skipped in order to create a dense reference model.
    # # This is then simply a replacement for xxd that removes a few manual steps.
    if args.numpy is not None:
        m.export_numpy(args.numpy)

    m.compress(args.method)
    m.store(args.output, args.cppmodel)


if __name__ == "__main__":
    cli()
