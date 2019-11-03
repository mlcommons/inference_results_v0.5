# SSD-ResNet34 Benchmark

## General Information

### Goal of this Benchmark

This benchmark performs object detection using SSD-ResNet34 network.

### Preprocessing

The input images are in INT8 NCHW format. Please refer to `scripts/preprocess_data.py` for more details about preprocessing.

### Model Source

The PyTorch model [resnet34-ssd1200.pytorch](resnet34-ssd1200.pytorch) is downloaded from the [zenodo link](https://zenodo.org/record/3236545/files/resnet34-ssd1200.pytorch) provided by the [MLPerf inference repository](https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection). We construct TensorRT network by reading layer and weight information from the PyTorch model. Details can be found in [SSDResNet34.py](SSDResNet34.py).

## Optimizations

### Plugins

The following TensorRT plugins were used to optimize SSDResNet34 benchmark:
- NMS_OPT_TRT: optimizes non-maximum suppression operation
The source codes of the plugins can be found in [../../plugins](../../plugins).

### Lower Precision

To further optimize performance, with minimal impact on classification accuracy, we run the computations in INT8 precision.

## Instructions for Audits

### Run Inference through LoadGen

Run the following commands from within the container to run inference through LoadGen:

```
make run RUN_ARGS="--benchmarks=ssd-large --scenarios=<SCENARIO> --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=ssd-large --scenarios=<SCENARIO> --test_mode=AccuracyOnly"
```

The performance and the accuracy results will be printed to stdout, and the LoadGen logs can be found in `build/logs`.

### Run with New Weights

Follow these steps to run inference with new weights:

1. Replace `build/models/SSDResNet34/resnet34-ssd1200.pytorch` with new PyTorch model.
2. Run `make calibrate RUN_ARGS="--benchmarks=ssd-large"` to generate a new calibration cache.
3. Run inference by `make run RUN_ARGS="--benchmarks=ssd-large --scenarios=<SCENARIO>"`.

### Run with New Validation Dataset

Follow these steps to run inference with new validation dataset:

1. Put the calibration dataset under `build/data/coco/val2017` and the new annotation data under `build/data/coco/annotations`.
2. Modify `data_maps/imagenet/val_map.txt` to contain all the file names of the new validation dataset according to their order in the annotation file.
3. Preprocess data by `python3 scripts/preprocess_data.py -d build/data -o build/preprocessed_data -b ssd-large --val_only`.
4. Run inference by `make run RUN_ARGS="--benchmarks=ssd-large --scenarios=<SCENARIO>"`.

### Run with New Calibration Dataset

Follow these steps to generate a new calibration cache with new calibration dataset:

1. Put the calibration dataset under `build/data/coco/train2017` and the new annotation data under `build/data/coco/annotations`.
2. Modify `data_maps/imagenet/cal_map.txt` to contain all the file names of the new calibration dataset.
3. Preprocess data by `python3 scripts/preprocess_data.py -d build/data -o build/preprocessed_data -b ssd-large --cal_only`.
4. Run `make calibrate RUN_ARGS="--benchmarks=ssd-large"` to generate a new calibration cache.
