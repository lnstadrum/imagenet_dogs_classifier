## Overview

This repository contains scripts to train a small neural net classifying images of dogs detailed here: [Towards GPU-accelerated image classification on low-end hardware](https://lnstadrum.medium.com/towards-gpu-accelerated-image-classification-on-low-end-hardware-ec592e125ad9)

The model of 216'152 trainable parameters achieves **72.28%** top-1 single crop validation accuracy on 385*385 pixels images of dogs and cats sampled from a challenging 120-class subset of ILSVRC 2012 dataset.

The model architecture is designed to be deployable for inference using an OpenGL ES 2.0-conformant GLSL implementation. This enables inference on a large spectrum of GPUs, from low-end and mobile devices to powerful desktop GPUs. The use of group convolutions, feature maps shuffling and a quantization-friendly activation function effectively reduces the memory bandwidth requirement making the architecture mobile-friendly. Desktop and Android test apps are available in [Beatmup](https://github.com/lnstadrum/beatmup).

TensorFlow/Keras is used for training.

## Installation

* Install cmake and gcc or clang

* Install [Beatmup Python package](https://github.com/lnstadrum/beatmup#quicker-start-with-python)

* Get the code, update submodules and build [FastAugment](https://github.com/lnstadrum/fastaugment) and sigmoid-like activation function extension for TensorFlow:

```bash
git submodule update --init --recursive
cd fastaugment
mkdir -p build && cd build
cmake .. && make
cd ../../sigmoid_like_tf_op
mkdir -p build && cd build
cmake .. && make
```

* Get ILSVRC 2012 dataset

* Run training/validation using [training.ipynb](blob/main/training.ipynb)

[beatmup_export.ipynb](blob/main/beatmup_export.ipynb) is provided to convert a trained model into a model that can be deployed for inference with Beatmup.