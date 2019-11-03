# GNMT

## Benckmark Information

https://github.com/mlperf/inference/blob/master/v0.5/translation/gnmt/tensorflow/README.md

## Model
### Publication/Attribution

```
@article{wu2016google,
  title={Google's neural machine translation system: Bridging the gap between human and machine translation},
  author={Wu, Yonghui and Schuster, Mike and Chen, Zhifeng and Le, Quoc V and Norouzi, Mohammad and Macherey, Wolfgang and Krikun, Maxim and Cao, Yuan and Gao, Qin and Macherey, Klaus and others},
  journal={arXiv preprint arXiv:1609.08144},
  year={2016}
}

```

## Dataset Preparation

### Publication/Attribution

BLEU evaluation is done on newstest2014 from WMT16 English-German
`@inproceedings{Sennrich2016EdinburghNM, title={Edinburgh Neural Machine
Translation Systems for WMT 16}, author={Rico Sennrich and Barry Haddow and
Alexandra Birch}, booktitle={WMT}, year={2016} }`

## Checkpoint

*   GNMT inference model loads checkpoint [from zenodo](https://zenodo.org/record/2530924/files/gnmt_model.zip).

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

### Installation of GCC

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

## GNMT Per-Configuration Settings

|benchmark|SUT|batch_size|Scenario|accuracy_mode|query_count|qps|time|
|-|-|-|-|-|-|-|-|
|gnmt|1x|16|Server|false|3903900|1400|120|
|gnmt|1x|128|Offline|false|3903900|4000|120|
|gnmt|2x|128|Offline|false|3903900|8000|120|
|gnmt|4x|128|Offline|false|3903900|16000|120|
|gnmt|8x|128|Offline|false|3903900|32000|120|
|gnmt|16x|128|Offline|false|3903900|64000|120|
