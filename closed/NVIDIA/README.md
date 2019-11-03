# MLPerf Inference v0.5 NVIDIA-Optimized Implementations

This is a repository of NVIDIA-optimized implementations for [MLPerf Inference benchmark v0.5](https://www.mlperf.org/inference-overview/).

## Benchmarks and Scenarios

As of MLPerf Inference v0.5, there are five *benchmarks*: **resnet**, **mobilenet**, **ssd-large**, **ssd-small**, and **gnmt**, and each benchmark runs in each of the four inference *scenarios*: **SingleStream**, **MultiStream**, **Offline**, and **Server**. Please refer to the `README.md` in [code](code) directory for detailed implementation and the optimizations done for each benchmark, and refer to [MLPerf Inference official page](https://www.mlperf.org/inference-overview/) for explanations about the scenarios.

## Support Matrix

Below shows the support matrix of each benchmark-scenario combination on each NVIDIA submission system:

|              | Tesla T4 x8                        | Tesla T4 x20                       | TITAN RTX x4                       | Jetson AGX Xavier<br>(w/ DLA x2)         |
|--------------|------------------------------------|------------------------------------|------------------------------------|------------------------------------------|
| resnet       | MultiStream<br>Offline<br>Server   | Offline<br>Server                  | MultiStream<br>Offline<br>Server   | SingleStream<br>MultiStream<br>Offline   |
| mobilenet    | MultiStream<br>Offline<br>Server   | X                                  | MultiStream<br>Offline<br>Server   | SingleStream<br>MultiStream<br>Offline   |
| ssd-large    | SingleStream<br>MultiStream<br>Offline<br>Server  | Offline<br>Server   | SingleStream<br>MultiStream<br>Offline<br>Server | SingleStream<br>MultiStream<br>Offline   |
| ssd-small    | MultiStream<br>Offline<br>Server   | Offline<br>Server                  | MultiStream<br>Offline<br>Server   | SingleStream<br>MultiStream<br>Offline   |
| gnmt         | SingleStream<br>Offline<br>Server  | Offline<br>Server                  | SingleStream<br>Offline<br>Server  | Please refer to [open division](../../open/NVIDIA)       |

## General Instructions

This section describes the steps needed to run harness with default configurations, weights, and validation datasets on NVIDIA submission systems to reproduce . Please refer to later sections for instructions for auditors.

### Prerequisites

Below are the prerequisites for x86_64 systems:
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
- NVIDIA GPUs with Turing architecture

Below are the prerequisites for Xavier:
- [NVIDIA JetPack 4.3 DP](https://developer.nvidia.com/jetpack-4_3_DP)
- Other dependencies can be installed by running this script: [install_xavier_dependencies.sh](scripts/install_xavier_dependencies.sh)

### Build Docker Image (x86_64 only)

Run the following command to build the docker image:

```
make build_docker
```

This command builds a docker image with the tag `mlperf-inference:<USERNAME>-latest`. The source codes in the repository are located at `/work` inside the docker image.

The commands in later sections assume that they are called within the docker container. Below shows an example of how to call commands to be executed in the container from outside the container:

```
docker run -dt -e NVIDIA_VISIBLE_DEVICES=ALL -w /work \
    --security-opt apparmor=unconfined --security-opt seccomp=unconfined \
    -v $HOME:/mnt$HOME \
    --name mlperf-inference-<USERNAME> mlperf-inference:<USERNAME>-latest
docker exec mlperf-inference-<USERNAME> bash -c '<COMMAND1>'
docker exec mlperf-inference-<USERNAME> bash -c '<COMMAND2>'
docker exec mlperf-inference-<USERNAME> bash -c '<COMMAND3>'
...
docker container stop mlperf-inference-<USERNAME>
docker container rm mlperf-inference-<USERNAME>
```

### Build Source Codes

Run the following command to build the source codes for GNMT benchmark, TensorRT plugins, and the harnesses:

```
make build
```

### Download and Preprocess Datasets

Download [ImageNet 2012 validation set](http://image-net.org/challenges/LSVRC/2012/) into `build/data/imagenet`, run the following commands to download the COCO datasets and the datasets needed for gnmt, and then preprocess the data.

```
make download_dataset
make preprocess_data
```

The downloaded and the preprocessed data will be saved in the following structure:

```
build
├── data
|   ├── coco
|   │   ├── annotations
|   │   ├── train2017
|   │   └── val2017
|   └── imagenet
└── preprocessed_data
    ├── coco
    │   ├── annotations
    │   ├── train2017
    │   │   ├── SSDMobileNet
    │   │   │   └── fp32
    │   │   └── SSDResNet34
    │   │       └── fp32
    │   └── val2017
    │       ├── SSDMobileNet
    │       │   ├── int8_chw4
    │       │   └── int8_linear
    │       └── SSDResNet34
    │           └── int8_linear
    └── imagenet
        ├── MobileNet
        │   ├── fp32
        │   └── int8_chw4
        └── ResNet50
            ├── fp32
            └── int8_linear
```

The preprocessed data are very large in size (~40GB). Please make sure you have enough space for storage.

### Download Benchmark Models

Run the following commands to download the models (weights):

```
make download_model
```

The models will be saved to `build/models`.

### Run Calibration

The calibration caches generated from default calibration set are already provided in each benchmark directory.

If you would like to re-generate the calibration cache for a specific benchmark, please run the following command:

```
make calibrate RUN_ARGS="--benchmarks=<BENCHMARK>"
```

### Generate TensorRT Optimized Plan Files

```
make generate_engines RUN_ARGS="--benchmarks=<BENCHMARK> --scenarios=<SCENARIO>"
```

The optimized plan files will be saved to `/work/build/engines`.

### Run Harness

Run the following command to launch the harness with optimized plan files and the LoadGen:

```
make run_harness RUN_ARGS="--benchmarks=<BENCHMARK> --scenarios=<SCENARIO> --test_mode=PerformanceOnly"
```

The performance results will be printed in stdout, and the LoadGen logs can be found in `/work/build/logs`.

Notes:
- To run AccuracyOnly mode, please use `--test_mode=AccuracyOnly` in `RUN_ARGS`.
- MultiStream scenario takes very long time to run (4-5 hours) due to the minimum query count requirement. If you would like to run it for shorter runtime, please add `--min_query_count=1` to `RUN_ARGS`.
- To achieve maximum performance, please set Transparent Huge Pages (THP) to *always* for image benchmarks in Server scenario.

### Update Results

Run the following command to update the LoadGen logs in `results/`:

```
make update_results
```

Please refer to [scripts/update_results.py](scripts/update_results.py) if the LoadGen logs are not in `build/logs`.

## Instructions for Auditors

Please refer to the `README.md` in each benchmark directory for auditing instructions.

## Appendix

### Flags for RUN_ARGS

- `--test_mode=[SubmissionRun,PerformanceOnly,AccuracyOnly]` specifies which LoadGen mode to run with
- `--verbose` prints out verbose logs
- `--benchmarks=comma,separated,list,of,benchmark,names`
- `--scenarios=comma,separated,list,of,scenario,names`
- `--no_gpu` disables generating / running GPU engine (Xavier only)
- `--gpu_only` disables generating / running DLA engine (Xavier only)
- `--force_calibration` forces recalculation of calibration cache
- `--log_dir=path/to/logs` species where to save logs
- `--log_copy_detail_to_stdout` prints LoadGen detailed logs to stdout as well as to the log files

### List of Make Targets

- `make prebuild` runs the following steps:
  - `make build_docker` builds docker image.
  - `make docker_add_user` adds current user to the docker image.
  - `make attach_docker` runs docker image with an interactive session and with current working directory bound to `/work` in the container.
- `make download_dataset` downloads datasets.
- `make preprocessed_data` preprocesses the downloaded datasets.
- `make build` runs the following steps:
  - `make clone_loadgen` clone the official MLPerf inference GitHub repo
  - `make build_gnmt` builds GNMT source codes.
  - `make build_plugins` builds TensorRT plugins.
  - `make build_loadgen` builds LoadGen source codes.
  - `make build_harness` builds the harnesses.
- `make run RUN_ARGS="--benchmarks=<BENCHMARK> --scenarios=<SCENARIO>"` runs the following steps:
  - `make generate_engines RUN_ARGS="--benchmarks=<BENCHMARK> --scenarios=<SCENARIO>"` generates TensorRT optimized plan files.
  - `make run_harness RUN_ARGS="--benchmarks=<BENCHMARK> --scenarios=<SCENARIO>"` runs harnesses with plan files and LoadGen.
- `make calibrate RUN_ARGS="--benchmarks=<BENCHMARK>"` generates calibration caches.
- `make clean` cleans up all the build directories. Please note that you will need to exit the docker container and run `make prebuild` again after the cleaning.
- `make clean_shallow` cleans up only the files needed to make a clean build.
- `make info` displays useful build information.
- `make shell` will spawn an interactive shell that inherits all the environment variables that the Makefile sets to provide an environment that mirrors the build/run environment.
