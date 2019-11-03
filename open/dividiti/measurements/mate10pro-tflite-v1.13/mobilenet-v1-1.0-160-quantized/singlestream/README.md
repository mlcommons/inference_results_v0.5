# MLPerf Inference - Open Division - Image Classification

We performed our measurements using automated, customizable, portable and reproducible
[Collective Knowledge](http://cknowledge.org) workflows. Our workflows automatically
install dependencies (models, datasets, etc.), preprocess input data in the correct way,
and so on.

## CK repositories

As CK is always evolving, it is hard to pin particular revisions of all repositories.

The most relevant repositories and their latest revisions on the submission date:
- [ck-mlperf](https://github.com/ctuning/ck-mlperf) @ [ee77cfd](https://github.com/ctuning/ck-mlperf/commit/ee77cfd3ddfa30739a8c2f483fe9ba83a233a000) (contains programs integrated with LoadGen, model packages and scripts).
- [ck-env](https://github.com/ctuning/ck-env) @ [f9ac337](https://github.com/ctuning/ck-env/commit/f9ac3372cdc82fa46b2839e45fc67848ab4bac03) (contains dataset descriptions, preprocessing methods, etc.)
- [ck-tensorflow](https://github.com/ctuning/ck-tensorflow) @ [eff8bec](https://github.com/ctuning/ck-tensorflow/commit/eff8bec192021162e4a336dbd3e795afa30b7d26) (contains TFLite packages).
- [armnn-mlperf](https://github.com/arm-software/armnn-mlperf) @ [42f44a2](https://github.com/ARM-software/armnn-mlperf/commit/42f44a266b6b4e04901255f46f6d34d12589208f) (contains ArmNN/ArmCL packages).

## Links
- [Bash script](https://github.com/ctuning/ck-mlperf/tree/master/script/mlperf-inference-v0.5.open.image-classification) used to invoke benchmarking on Linux systems or Android devices.
