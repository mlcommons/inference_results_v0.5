# SSD Large

## Benckmark Information

SSD Large is
[ssd-resnet34 1200x1200](https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection#mlperf-inference-benchmarks-for-image-classification-and-object-detection-tasks)
benckmark.

## Model
### Publication/Attribution

Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector. In the
Proceedings of the European Conference on Computer Vision (ECCV), 2016.

## Dataset Preparation

Microsoft COCO: COmmon Objects in Context. 2017.

*   [SSD COCO Dataset preparation](https://github.com/tensorflow/tpu/tree/master/models/official/retinanet#preparing-the-coco-dataset)

## Checkpoint

*   SSD Large inference model loads checkpoint [from zenodo](https://zenodo.org/record/3345892/files/tf_ssd_resnet34_22.1.zip?download=1).

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
installed and the `PYTHONPATH` environment variable is not set. All submissions
are run using Python3.

### Installation of GCC.

*   This is not necessary to run the models, but it is needed to run the
    [Cloud TPU Profiler](https://cloud.google.com/tpu/docs/cloud-tpu-tools#profile_tab)
    which can be used to extract performance information from the TPU. However,
    feel free to replace the `upgrade_gcc.sh` script packaged in this submission
    with an empty script.


## Research submissions

A subset of research submissions were run using Google internal
infrastructure. Contact Peter Mattson (petermattson@google.com) for more
details.

Here is a table of internal code revisions that were used for each model.

GNMT: cl/274015478
SSD: cl/273348435
ResNet: cl/274032990

## SSD Per-Configuration Settings

|benchmark|SUT|batch_size|batch_timeout_micros|count|max_latency|scenario|qps|init_iterations|use_bfloat16|threads|use_space_to_depth|accuracy|
|-|-|-|-|-|-|-|-|-|-|-|-|-|
|ssd|1x|4|10000|64|0.1|Server|320|32|false|128|true|false|
|ssd|1x|128|20000|64|100|Offline|700|32|true|96|false|false|
|ssd|2x|128|20000|64|100|Offline|1400|32|true|192|false|false|
|ssd|4x|128|20000|64|100|Offline|2800|32|true|384|false|false|
|ssd|8x|128|20000|64|100|Offline|5600|32|true|768|false|false|
|ssd|16x|128|20000|64|100|Offline|11200|32|true|1000|false|false|
