# SSD-MobileNet Benchmark

## General Information

### Goal of this Benchmark

This benchmark performs object detection using SSD-MobileNet network.

### Preprocessing

The input images are in INT8 NCHW format for T4x20 and INT8 NHW4 format for the other systems. Please refer to `scripts/preprocess_data.py` for more details about preprocessing.

### Model Source

The TensorFlow frozen graph [ssd_mobilenet_v1_coco_2018_01_28.pb](ssd_mobilenet_v1_coco_2018_01_28.pb) is downloaded from the [zenodo link](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) provided by the [MLPerf inference repository](https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection). We convert the TensorFlow frozen graph to UFF format, and then use TensorRT UFF parser with post-processing steps to convert the UFF model to TensorRT network. Details can be found in [SSDMobileNet.py](SSDMobileNet.py).

## Optimizations

### Plugins

The following TensorRT plugins were used to optimize SSDMobileNet benchmark:
- GridAnchor_TRT: generates grid anchors
- NMS_OPT_TRT: optimizes non-maximum suppression operation
The first plugin is available in [TensorRT 6](https://developer.nvidia.com/tensorrt) release. The source codes of the other plugins can be found in [../../plugins](../../plugins).

### Lower Precision

To further optimize performance, with minimal impact on classification accuracy, we run the computations in INT8 precision.

### Replace ReLU6 with ReLU

On DLA, we replace ReLU6 with ReLU to achieve further performance.

## Instructions for Audits

### Run Inference through LoadGen

Run the following commands from within the container to run inference through LoadGen:

```
make run RUN_ARGS="--benchmarks=ssd-small --scenarios=<SCENARIO> --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=ssd-small --scenarios=<SCENARIO> --test_mode=AccuracyOnly"
```

The performance and the accuracy results will be printed to stdout, and the LoadGen logs can be found in `build/logs`.

### Run with New Weights

Follow these steps to run inference with new weights:

1. Replace `build/models/SSDMobileNet/frozen_inference_graph.pb` with new TensorFlow frozen graph.
2. Run `make calibrate RUN_ARGS="--benchmarks=ssd-small"` to generate a new calibration cache.
3. Run inference by `make run RUN_ARGS="--benchmarks=ssd-small --scenarios=<SCENARIO>"`.

### Run with New Validation Dataset

Follow these steps to run inference with new validation dataset:

1. Put the calibration dataset under `build/data/coco/val2017` and the new annotation data under `build/data/coco/annotations`.
2. Modify `data_maps/imagenet/val_map.txt` to contain all the file names of the new validation dataset according to their order in the annotation file.
3. Preprocess data by `python3 scripts/preprocess_data.py -d build/data -o build/preprocessed_data -b ssd-small --val_only`.
4. Run inference by `make run RUN_ARGS="--benchmarks=ssd-small --scenarios=<SCENARIO>"`.

### Run with New Calibration Dataset

Follow these steps to generate a new calibration cache with new calibration dataset:

1. Put the calibration dataset under `build/data/coco/train2017` and the new annotation data under `build/data/coco/annotations`.
2. Modify `data_maps/imagenet/cal_map.txt` to contain all the file names of the new calibration dataset.
3. Preprocess data by `python3 scripts/preprocess_data.py -d build/data -o build/preprocessed_data -b ssd-small --cal_only`.
4. Run `make calibrate RUN_ARGS="--benchmarks=ssd-small"` to generate a new calibration cache.
