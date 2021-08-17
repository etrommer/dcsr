# Delta-Compressed Storage Row
Software stack for the paper "dCSR: A Memory-Efficient Sparse Matrix Representation for Parallel Neural Network Inference"

## Key components
These are the most important parts of this repository:

- `training` Training and pruning scripts for Keyword spotting and MobileNetV2 reference models
- `models` Reference models that can be used when wanting to skip the training step
- `conversion` Utilities to convert arrays and TFlite models to dCSR
  - `conversion/compress_tflite.py`: Identify compressible tensors in a .tflite model and convert to dCSR
  - `conversion/generate_dcsr_ref.py`: Turn a .npy file into a header that can be compiled into the sparse kernel test harness. Target tensors can also be exported from an existing model with the help of the `compress_tflite.py` script.
- `m55_inference` Adapted Tensorflow lite micro runtime with custom kernels to run dCSR-compressed models. Models can either be simulated on the [ARM Corstone-300 FVP](https://developer.arm.com/tools-and-software/open-source-software/arm-platforms-software/arm-ecosystem-fvps). This does only allow for functional testing and will not give you performance benchmarks (it actually does report cycle counts, but they are not accurate enough to serve as a benchmark) or the [Arm MPS3 FPGA protoyping board](https://developer.arm.com/tools-and-software/development-boards/fpga-prototyping-boards/mps3). This can be used for profiling throughput as well, but you will need to buy one...
  - `m55_inference/test_files`: Testing harness for sparse kernels. This is particularly helpful for a quick test as well as tracking down errors. Test data can be generated from any Numpy array using the `repacking_utils/generate_dcsr_ref.py`. The tests are compiled separately and do not invoke the Tensorflow lite micro runtime.
  - `m55_inference/sparse_kernels`: Kernel functions for dCSR inference
  - `m55_inference/tensorflow` Adapted TFlite micro runtime that adds support for additional sparsity information required by dCSR. Normally, any sparsity information in the model is ignored. The adapted version reads it and forwards it to the kernel invocation.

## Prerequisites
### Training/Conversion
- Install Python dependencies: `pip install -r requirements.txt`

### Building TFLite firmware
- Download and unpack the [ARM GNU toolchain](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm/downloads) (tested with `gcc-arm-none-eabi-10-2020-q4-major`)
- Set environment variable `ARM_GCC_DIR` to point to it:
```export ARM_GCC_DIR="<your path to ARM GCC>"```
- Download [CMSIS5](https://github.com/ARM-software/CMSIS_5) and set environment variable `CMSIS_DIR` to point to it:
```export CMSIS_DIR="<your path to CMSIS>"```

### Simulating firmware on the Corstone300 FVP
- Download and install the [Corstone-300 FVP](https://developer.arm.com/tools-and-software/open-source-software/arm-platforms-software/arm-ecosystem-fvps)

- Make sure the Corstone-300 executable is in your `$PATH`:

  ```export PATH="<Corstone 300 Directory>:$PATH"```

## Use cases
These are the most common commands and can be used as a starting point. Conversion scripts support the `-h`  flag in order to get more information on the individual options.

### Train a model
```bash
cd training
python train_kws.py
```

### Convert a TFlite model
This takes a vanilla TFlite model, identifies sparse tensors, converts them to TFlite and writes the result to a source file in the TFlite micro file structure from where it can be compiled directly.

```bash
cd conversion
python compress_tflite.py \
	../models/kws_networks/sparse_s_80.tflite \
	--compress \
	--cppmodel ../m55_inference/tensorflow/lite/micro/examples/hello_world/model.cc \
	--debug
```

### Run a TFlite model
After converting a model, it can be run in one of two ways:

- Run on Corstone-300 simulator:

  ```bash
  cd m55_inference
  make sim
  ```

- Build for MPS3
  ```bash
  cd m55_inference
  make all FPGA=1
  ```

### Create a dCSR test source from a .npy file
In case you want to experiment with dCSR and the sparse kernels directly, without the overhead of TFlite micro, all you need is a sparse array that you would like to convert in `.npy` format.

It can be converted to a C source file that can be compiled together with the test environment like this:

```bash
cd conversion
python generate_dcsr_ref.py <your array>.npy ../m55_inference/test_files/test_matrix.c
```

### Run test
After creating your test source, you can compile and run them like this:

- Run on Corstone-300 simulator:
  ```bash
  cd m55_inference
  make sim TEST=1
  ```
- Build for MPS3
  ```bash
  cd m55_inference
  make all TEST=1 FPGA=1
  ```
