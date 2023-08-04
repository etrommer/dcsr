<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="dcsr_logo.png" alt="Logo" width="400" >
  </a>

  <p align="center">
    Software implementation for the paper <a href = "https://arxiv.org/abs/2111.12345">dCSR: A Memory-Efficient Sparse Matrix Representation for Parallel Neural Network Inference</a>
    <br>
    <br>
    <img alt="Static Badge" src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
    <img alt="Static Badge" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white">
    <img alt="Static Badge" src="https://img.shields.io/badge/MLFlow-0194E2?style=for-the-badge&logo=MLFlow&logoColor=white">
    <img alt="Static Badge" src="https://img.shields.io/badge/ARM_Cortex--M55-0091BD?style=for-the-badge&logo=arm&logoColor=white">
    <br>
    <a href="https://github.com/etrommer/dcsr/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/etrommer/dcsr?style=for-the-badge&logo=github"></a>
  </p>
</div>


# Delta-Compressed Storage Row
dCSR is compression format for sparse weight matrices in Neural Networks. This repository implements these steps:
1. Pruning using TensorFlow/Tensorflow Datasets/Tensorflow Model Optimization Toolkit. Two reference architectures are available: A [Depthwise Convolutional Neural Network for Keyword Spotting](https://arxiv.org/abs/1711.07128) and [MobileNetV2 on the CIFAR10 dataset](https://arxiv.org/abs/1801.04381). Training and pruning is tracked using MLFlow. Models are converted to Int8 quantization and exported using Tensorflow lite.
2. In-place conversion of the `.tflite` model. A script reads in a Tensorflow lite model, identifies all sparse weight tensors and encodes them using dCSR or the reference Run-Length Encoding.
3. Inference on the [ARM-MPS3 AN547 Hardware Prototype](https://developer.arm.com/Tools%20and%20Software/MPS3%20FPGA%20Prototyping%20Board). At the time of writing, this was the only available implementation of the ARM Cortex-M55 MCU that implements the ARM M-Profile Vector Extension (also known as "Helium Instructions")

# Usage
## Installation - Python
For simply converting sparse Numpy matrices to dCSR and measuring compression rates, install this package using pip:
```bash
pip install git+https://github.com/etrommer/dcsr.git
```
For training, pruning and converting models, follow these steps:
This project uses [poetry](https://python-poetry.org/docs/) for dependency management. Make sure that it is installed on your system.
Then clone and install the project with the optional dependencies for training and pruning Neural networks
```bash
git clone https://github.com/etrommer/dcsr.git
cd dcsr
poetry install --with training
```
## Installation - C
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
## Training
There are two entry points, corresponding to the two experiments from the paper.
For MobileNetV2 and CIFAR-10, run `poetry run train_mnetv2`:
```bash
> poetry run train_mnetv2 --help

usage: train_mnetv2 [-h] [-e EPOCHS] [-p PRUNING_EPOCHS] [-r RETRAINING_EPOCHS] [-o DROPOUT] [-w WIDTH] [-s SPARSITY] [-x SKIP_BLOCKS] [-b BASE_MODEL_PATH] [-d]

Training and Pruning for Keyword Spotting DS-CNN

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        Number of Training Epochs for Dense Base Model
  -p PRUNING_EPOCHS, --pruning_epochs PRUNING_EPOCHS
                        Number of Pruning Epochs
  -r RETRAINING_EPOCHS, --retraining_epochs RETRAINING_EPOCHS
                        Number of Retraining Epochs after Pruning
  -o DROPOUT, --dropout DROPOUT
                        Amount of Dropout to add to MobileNetV2 to prevent overfitting
  -w WIDTH, --width WIDTH
                        Width Multiplier for MobileNetV2
  -s SPARSITY, --sparsity SPARSITY
                        Sparsity to prune to
  -x SKIP_BLOCKS, --skip_blocks SKIP_BLOCKS
                        Number of initial inverted residual blocks that remain dense
  -b BASE_MODEL_PATH, --base_model_path BASE_MODEL_PATH
                        Pre-trained dense base model to prune
  -d, --debug           Print Debug Output
```
For the Depthwise-Separable CNN on the Google Speech commands dataset, use `poetry run train_kws`:
```bash
> poetry run train_kws --help

usage: train_kws [-h] [-e EPOCHS] [-p PRUNING_EPOCHS] [-r RETRAINING_EPOCHS] [-a {s,m,l}] [-w WEIGHTS] [-d]

Training and Pruning for Keyword Spotting DS-CNN

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        Number of Training Epochs for Dense Base Model
  -p PRUNING_EPOCHS, --pruning_epochs PRUNING_EPOCHS
                        Number of Pruning Epochs
  -r RETRAINING_EPOCHS, --retraining_epochs RETRAINING_EPOCHS
                        Number of Retraining Epochs after Pruning
  -a {s,m,l}, --architecture {s,m,l}
                        Model Architecture as described in 'Hello Edge' paper
  -w WEIGHTS, --weights WEIGHTS
                        Path to weights for Dense Base Network.Weights need to be generated by the same architecture as given in the 'architecture' parameter.Providing weights skips the initial training of a dense base model.
  -d, --debug           Print Debug Output
```
## Model conversion
Use the provided `compress_tflite` script: `poetry run compress_tflite --output <output_model_path>.tflite -m dcsr <uncompressed input model>`. 
Different compression methods are supported using the `-m` flag. The resulting can either be written to another `.tflite` file or directly converted to a C array definition in the format that is expected the Tensorflow lite micro inference environment.

## Pre-trained models
The sparse models evaluated in the paper can be found in the `models` directory

# Inference
**Warning**: As of summer 2023, this part has been deprecated. Keeping a TFlite micro project with custom kernels up-to-date with upstream dependencies is not trivial and would be too time-consuming for limited benefit. The inference code is kept for reference under `firmware`. Most components _should_ still work with minor adjustments. 

## Key components
This is how the inference portion of this repository is generally organized:
- `firmware` Adapted Tensorflow lite micro runtime with custom kernels to run dCSR-compressed models. Models can either be simulated on the [ARM Corstone-300 FVP](https://developer.arm.com/tools-and-software/open-source-software/arm-platforms-software/arm-ecosystem-fvps). This does only allow for functional testing and will not give you performance benchmarks (it actually does report cycle counts, but they are not accurate enough to serve as a benchmark) or the [Arm MPS3 FPGA protoyping board](https://developer.arm.com/tools-and-software/development-boards/fpga-prototyping-boards/mps3). This can be used for profiling throughput as well, but you will need to buy one...
  - `firmware/test_files`: Testing harness for sparse kernels. This is particularly helpful for a quick test as well as tracking down errors. Test data can be generated from any Numpy array using the `numpy_test` script. The tests are compiled separately and do not invoke the Tensorflow lite micro runtime.
  - `firmware/sparse_kernels`: Kernel functions for dCSR inference
  - `firmware/tensorflow` Adapted TFlite micro runtime that adds support for additional sparsity information required by dCSR. Normally, any sparsity information in the model is ignored. The adapted version reads it and forwards it to the kernel invocation.

## Running a TFlite model
After converting a model, it can be run in one of two ways (enter the firmware directory using `cd firmware` before running these)

- Run on Corstone-300 simulator:
  ```bash
  make sim
  ```

- Build for MPS3
  ```bash
  make all FPGA=1
  ```
