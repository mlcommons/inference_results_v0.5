# MLPerf Inference - Open Division - Object Detection

We performed our measurements using automated, customizable, portable and reproducible
[Collective Knowledge](http://cknowledge.org) workflows. Our workflows automatically
install dependencies (models, datasets, etc.), preprocess input data in the correct way,
and so on.

## CK repositories

As CK is always evolving, it is hard to pin particular revisions of all repositories.

The most relevant repositories and their latest revisions on the submission date (18/Oct/2019):

- [ck-mlperf](https://github.com/ctuning/ck-mlperf) @ [ef1fced](https://github.com/ctuning/ck-mlperf/commit/ef1fcedd495fd03b5ad6d62d62c8ba271854f2ad) (contains the CK program wrapper, MLPerf SSD-MobileNet model packages and scripts).
- [ck-object-detection](https://github.com/ctuning/ck-object-detection) @ [780d328](https://github.com/ctuning/ck-object-detection/commit/780d3288ec19656cb60c5ad39b2486bbf0fbf97a) (contains most model packages)
- [ck-env](https://github.com/ctuning/ck-env) @ [5af9fbd](https://github.com/ctuning/ck-env/commit/5af9fbd93ad6c6465b631716645ad9442a333442) (contains dataset descriptions, preprocessing methods, etc.)

## Links
- [Docker image with instructions](https://github.com/ctuning/ck-mlperf/tree/master/docker/mlperf-inference-vision-with-ck.tensorrt.ubuntu-18.04).
- [Bash script](https://github.com/ctuning/ck-mlperf/tree/master/script/mlperf-inference-v0.5.open.object-detection) used to invoke benchmarking via the Docker image.
