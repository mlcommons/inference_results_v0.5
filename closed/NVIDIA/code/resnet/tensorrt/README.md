# ResNet50 Benchmark

## General Information

### Goal of this Benchmark

This benchmark performs image classification using ResNet-50 network.

### Preprocessing

The input images are in INT8 NCHW format. Please refer to `scripts/preprocess_data.py` for more details about preprocessing.

### Model Source

The ONNX model *resnet50_v1.onnx* is downloaded from the [zenodo link](https://zenodo.org/record/2592612/files/resnet50_v1.onnx) provided by the [MLPerf inference repository](https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection). We then use TensorRT ONNX parser with post-processing steps to convert the ONNX model to TensorRT network. Details can be found in [ResNet50.py](ResNet50.py).

## Optimizations

### Plugins

The following TensorRT plugins were used to optimize ResNet50 benchmark:
- RnRes2Br1Br2c_TRT: fuses res2a_br1 and res2a_br2c layers into one CUDA kernel
- RnRes2Br2bBr2c_TRT: fuses res2b_br2b and res2b_br2c or res2c_br2b and res2c_br2c layers into one CUDA kernel
These plugins are available in [TensorRT 6](https://developer.nvidia.com/tensorrt) release.

### Lower Precision

To further optimize performance, with minimal impact on classification accuracy, we run the computations in INT8 precision.

### Removal of Softmax

Softmax layer is removed since it does not affect the predicted label.

## Instructions for Audits

### Run Inference through LoadGen

Run the following commands from within the container to run inference through LoadGen:

```
make run RUN_ARGS="--benchmarks=resnet --scenarios=<SCENARIO> --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=resnet --scenarios=<SCENARIO> --test_mode=AccuracyOnly"
```

The performance and the accuracy results will be printed to stdout, and the LoadGen logs can be found in `build/logs`.

### Run with New Weights

Follow these steps to run inference with new weights:

1. If the new weights are in TensorFlow frozen graph format, please use [resnet50-to-onnx.sh](https://github.com/mlperf/inference/blob/master/v0.5/classification_and_detection/tools/resnet50-to-onnx.sh) in the official MLPerf repository to convert it to ONNX format.
2. Replace `build/models/ResNet50/resnet50_v1.onnx` with new ONNX model.
3. Run `make calibrate RUN_ARGS="--benchmarks=resnet"` to generate a new calibration cache.
4. Run inference by `make run RUN_ARGS="--benchmarks=resnet --scenarios=<SCENARIO>"`.

### Run with New Validation Dataset

Follow these steps to run inference with new validation dataset:

1. Put the calibration dataset under `build/data/imagenet`.
2. Modify `data_maps/imagenet/val_map.txt` to contain all the file names and the corresponding labels of the new validation dataset.
3. Preprocess data by `python3 scripts/preprocess_data.py -d build/data -o build/preprocessed_data -b resnet --val_only`.
4. Run inference by `make run RUN_ARGS="--benchmarks=resnet --scenarios=<SCENARIO>"`.

### Run with New Calibration Dataset

Follow these steps to generate a new calibration cache with new calibration dataset:

1. Put the calibration dataset under `build/data/imagenet`.
2. Modify `data_maps/imagenet/cal_map.txt` to contain all the file names of the new calibration dataset.
3. Preprocess data by `python3 scripts/preprocess_data.py -d build/data -o build/preprocessed_data -b resnet --cal_only`.
4. Run `make calibrate RUN_ARGS="--benchmarks=resnet"` to generate a new calibration cache.
