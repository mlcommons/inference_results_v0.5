# GNMT Benchmark

## General Information

### Goal of this Benchmark
This benchmark performs translation from one natural language to another, such as English to German. As explained below, it implements Google's Neural Machine Translation (GNMT) system. Aside from using a smaller selection of plugins, this submission is identical to NVIDIA's GNMT closed division submission. It was not submitted there due to administrative error.

### Preprocessing
The input to this benchmark is a preprocessed text file. The preprocessing steps consist of running tokenization and splitting into subwords, using byte pair encoding (bpe).

These steps can be performed by
1.  Downloading the [bpe file](https://github.com/mlperf/inference/blob/master/v0.5/translation/gnmt/tensorflow/download_dataset.sh)
2.  Running the [preprocessing script](https://github.com/mlperf/inference/blob/master/v0.5/translation/gnmt/tensorflow/preprocess_input.sh)

### Model Source
We construct the network for this benchmark using [TensorRT](https://developer.nvidia.com/tensorrt) APIs and load the model weights from the official [GNMT TensorFlow model](https://github.com/mlperf/inference/tree/master/v0.5/translation/gnmt/tensorflow) from this [zenodo link](https://zenodo.org/record/2530924/files/gnmt_model.zip). Our network structure follows the original network structure of the model file.

## Optimizations
### Plugins
The following custom CUDA-code plugins were used to optimize GNMT:
*  SingleStepLSTM_TRT: for the decoder running in FP16
*  AttentionPlugin: fusing several layers of the attention module
*  ScorerPlugin: fusing several layers of the scorer
*  MultiGatherPlugin: implements efficient shuffling of hidden states for beam search

The first plugin is available in [TensorRT 6](https://developer.nvidia.com/tensorrt) release. The source codes of the other plugins can be found in [src/plugin](src/plugin).

### Lower Precision
To further optimize performance, with minimal impact on translation accuracy, precision of internal tensors can be lowered.

#### FP16 Precision
GNMT can be run in FP16 precision, instead of FP32, using the following command line argument:
```
-t fp16
```

#### Int8 Precision
In addition to running the GNMT network in FP16 precision, parts of the network, the projection layer in the scorer, also support Int8 precision. This can be activated using the following argument:
```
-t fp16 --int8Generator
```

## Instructions for Audits
### Prerequisites
To run any of the instructions below, ensure that you have:
*  A vocabulary file, [e.g., the one from the official GitHub](https://github.com/mlperf/inference/blob/master/v0.5/translation/gnmt/tensorflow/download_dataset.sh)
*  A TensorFlow checkpoint, [e.g., the one from the official GitHub](https://github.com/mlperf/inference/blob/master/v0.5/translation/gnmt/tensorflow/download_trained_model.sh)
*  A preprocessed input file, or a script to preprocess raw input text. See [Preprocessing](#preprocessing)

Note that these three items are all interdependent: GNMT is trained according to a specific vocabulary file, hence the weights are dependent on the vocabulary. Likewise, the input needs to be split into sub-words that appear in the vocabulary.
For the remainder of this document, we will assume a preprocessed input file newstest2014.tok.bpe.32000.en exists and is located in $DATA/newstest2014.tok.bpe.32000.en.

### Run Inference through LoadGen
Ensure you have generated weights and a calibration cache before continuing (see the corresponding sections below).

Run the following commands from within the container to run inference through LoadGen:

```
make run RUN_ARGS="--benchmarks=gnmt --scenarios=<SCENARIO> --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=gnmt --scenarios=<SCENARIO> --test_mode=AccuracyOnly"
```

The performance and the accuracy results will be printed to stdout, and the LoadGen logs can be found in `build/logs`.

### Generate New Weights
From a pretrained TensorFlow model, run the following on a checkpoint

```
./code/gnmt/tensorrt/convertTFWeights.py -m <TensorFlow checkpoint basename> -o build/models/GNMT/gnmt.wts
```

e.g.,

```
./code/gnmt/tensorrt/convertTFWeights.py -m translate.cpkt -o build/models/GNMT/gnmt.wts
```

### Generate Calibration Caches 
To be able to use the int8 precision optimization, a calibration cache needs to be generated first. This can be done as follows:
```
make calibrate RUN_ARGS="--benchmarks=gnmt"
```

More details on the steps that this make target performs can be found in the calibrate function in [GNMT.py](GNMT.py).
The resulting calibration cache will be stored in [data/Int8CalibrationCache](data/Int8CalibrationCache). 
This calibration cache can now be used through the "--calibration_cache" command line option, e.g.,:

```
./build/bin/GNMT/gnmt -t fp16 --int8Generator --input_file $DATA/newstest2014.tok.bpe.32000.en --bs 128 --bm 10 --calibration_cache myCalibrationCache
```

### GNMT Test Modes
Ensure you have generated weights before continuing.

To test GNMT in FP32 precision, the following command can be ran:
```
./build/bin/GNMT/gnmt --input_file $DATA/newstest2014.tok.bpe.32000.en --bs 128 --bm 10
```

To test GNMT in FP16 precision, please add the "-t fp16" flags like below:
```
./build/bin/GNMT/gnmt -t fp16 --input_file $DATA/newstest2014.tok.bpe.32000.en --bs 128 --bm 10
```

If you have generated a calibration cache, myCalibrationCache, you can run GNMT with several of its modules in INT8 precision. Adding the "--int8Generator" to the previous command accomplishes this:
```
./build/bin/GNMT/gnmt -t fp16 --int8Generator --input_file $DATA/newstest2014.tok.bpe.32000.en --bs 128 --bm 10 --calibration_cache myCalibrationCache
```
