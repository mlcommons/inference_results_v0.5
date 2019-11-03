# ResNet-50

## Benckmark Information

MLPerf Inference v0.5 ResNet-50 is defined in
[resnet50-v1.5](https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection).

## Model.
### Publication/Attribution

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. [Deep Residual Learning for
Image Recognition.](https://arxiv.org/abs/1512.03385), 2015.

## Dataset Preparation

### Publiction/Attribution

ImageNet2012

* [imagenet2012 (validation)](http://image-net.org/challenges/LSVRC/2012/) https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection#datasets


## Checkpoint

*   ResNet-50 model checkpoint [from zenodo](https://zenodo.org/record/2535873/files/resnet50_v1.pb).

## Setup

## Cloud Environment

In order to run with a cloud TPU, you will need to
[configure your environment](https://cloud.google.com/tpu/docs/quickstart). In
particular, submissions assume that you have:

1.  In particular, it assumes that you have a
    [GCS storage bucket](https://cloud.google.com/tpu/docs/storage-buckets)
    which is located in the same region as your cloud TPU. The TPU's service
    account must access in order to both read the input data and write model
    checkpoints.
2.  The user instance must have
    [permissions](https://cloud.google.com/iam/docs/overview) to access cloud
    TPU APIs.
3.  The project must have [quota](https://cloud.google.com/storage/quotas) to
    create cloud TPUs for the submission.

## Local Environment

This README assumes a clean Ubuntu 18.04 instance running in Google Cloud. When
the submission is run it will perform a variety of system configurations; the
most notable are: 1. Construction of a Virtualenv. - The model will run using
this environment, and it is assumed that no other Python packages have been
installed and the `PYTHONPATH` environment variable is not set. ResNet
submissions use both Python3 and c++. The python script performs dataset
preprocessings and graph generations, and the c++ counterpart loads the graph
and preprocessed images and runs load tests. 2. See below for c++ compilation.

### Installation of GCC

To use the c++ part, one needs to install gcc-8 and g++-8 or beyond on the local
environment. For compilation, we use open-source [bazel](https://bazel.build/)
and provide a bazel BUILD file in the `resnet/cc` directory. Also, the c++ code
is compatible with c++-14 and beyond. To compile, run the following
command:

```
bazel build -c opt /path/to/resnet/cc:main --cxxopt="--std=c++14"
```


## Research submissions

Research submissions were run using Google internal infrastructure. Contact for
more information.

## ResNet Per-Configuration Settings

|benchmark|SUT|total_sample_count|scenario|preprocessing_and_graph_only|batch_size|qps|time|max_latency|space_to_depth_block_size|num_worker_threads|batching_batch_timeout_micros|batching_max_enqueued_batches|batching_num_batch_threads|accuracy_mode|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|resnet|1x|50000|Server|true|16|16000|60|15|2|800|1500|2097152|64|false|
|resnet|1x|50000|Offline|true|256|40000|60|1000|0|200|10|2097152|128|false|
|resnet|2x|50000|Offline|true|256|80000|60|1000|0|400|10|2097152|128|false|
|resnet|4x|50000|Offline|true|256|160000|60|1000|0|800|10|2097152|128|false|
|resnet|8x|50000|Offline|true|256|320000|60|1000|0|800|10|2097152|128|false|
|resnet|16x|50000|Offline|true|256|640000|60|1000|0|800|10|2097152|128|false|
|resnet|32x|50000|Offline|true|256|1280000|60|1000|0|800|10|2097152|128|false|
