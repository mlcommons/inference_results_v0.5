# MobileNet Benchmark

## General Information

### Goal of this Benchmark

This benchmark performs image classification using MobileNet network.

### Preprocessing

The input images are in INT8 NHW4 format. Please refer to `scripts/preprocess_data.py` for more details about preprocessing.

### Model Source

The prequantized ONNX model [mobilenet_sym_no_bn.onnx](mobilenet_sym_no_bn.onnx) is downloaded from the [zenodo link](https://zenodo.org/record/3353417/files/Quantized%20MobileNet.zip) provided by the [MLPerf inference repository](https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection). We use TensorRT ONNX parser with post-processing steps to convert the ONNX model to TensorRT network. Details can be found in [MobileNet.py](MobileNet.py).

## Optimizations

### Lower Precision

To optimize performance, with minimal impact on classification accuracy, we run the computations in INT8 precision.

### Removal of Softmax

Softmax layer is removed since it does not affect the predicted label.

## Instructions for Audits

### Run Inference through LoadGen

Run the following commands from within the container to run inference through LoadGen:

```
make run RUN_ARGS="--benchmarks=mobilenet --scenarios=<SCENARIO> --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=mobilenet --scenarios=<SCENARIO> --test_mode=AccuracyOnly"
```

The performance and the accuracy results will be printed to stdout, and the LoadGen logs can be found in `build/logs`.

### Run with New Weights

Follow these steps to run inference with new weights:

1. Replace `build/models/MobileNet/mobilenet_sym_no_bn.onnx` with new prequantized ONNX model.
2. Run `make calibrate RUN_ARGS="--benchmarks=mobilenet"` to generate a new calibration cache.
3. Run inference by `make run RUN_ARGS="--benchmarks=mobilenet --scenarios=<SCENARIO>"`.

### Run with New Validation Dataset

Follow these steps to run inference with new validation dataset:

1. Put the calibration dataset under `build/data/imagenet`.
2. Modify `data_maps/imagenet/val_map.txt` to contain all the file names and the corresponding labels of the new validation dataset.
3. Preprocess data by `python3 scripts/preprocess_data.py -d build/data -o build/preprocessed_data -b mobilenet --val_only`.
4. Run inference by `make run RUN_ARGS="--benchmarks=mobilenet --scenarios=<SCENARIO>"`.

### Run with New Calibration Dataset

Follow these steps to generate a new calibration cache with new calibration dataset:

1. Put the calibration dataset under `build/data/imagenet`.
2. Modify `data_maps/imagenet/cal_map.txt` to contain all the file names of the new calibration dataset.
3. Preprocess data by `python3 scripts/preprocess_data.py -d build/data -o build/preprocessed_data -b mobilenet --cal_only`.
4. Run `make calibrate RUN_ARGS="--benchmarks=mobilenet"` to generate a new calibration cache.
