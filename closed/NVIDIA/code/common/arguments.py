# Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse

arguments_dict = {
    # Common arguments
    "gpu_batch_size": {
        "help": "GPU batch size to use for the engine.",
        "type": int,
    },
    "dla_batch_size": {
        "help": "DLA batch size to use for the engine.",
        "type": int,
    },
    "batch_size": {
        "help": "Batch size to use for the engine.",
        "type": int,
    },
    "batch_sizes": {
        "help": "User-provided comma-separated list of batch sizes to use for server scenario.",
    },
    "verbose": {
        "help": "Use verbose output",
        "action": "store_true",
    },

    # Dataset location
    "data_dir": {
        "help": "Directory containing unprocessed datasets",
        "default": os.environ.get("DATA_DIR", "build/data"),
    },
    "preprocessed_data_dir": {
        "help": "Directory containing preprocessed datasets",
        "default": os.environ.get("PREPROCESSED_DATA_DIR", "build/preprocessed_data"),
    },

    # Arguments related to precision
    "precision": {
        "help": "Precision. Default: int8",
        "choices": ["fp32", "fp16", "int8", None],
        # None needs to be set as default since passthrough arguments will
        # contain a default value and override configs. Specifying None as the
        # default will cause it to not be inserted into passthrough / override
        # arguments.
        "default": None,
    },
    "explicit_precision": {
        "help": "Use explicit precision mode (only for MobileNet).",
        "action": "store_true",
    },
    "input_dtype": {
        "help": "Input datatype. Choices: fp32, int8.",
        "choices": ["fp32", "int8", None],
        "default": None
    },
    "input_format": {
        "help": "Input format (layout). Choices: linear, chw4",
        "choices": ["linear", "chw4", None],
        "default": None
    },

    # Arguments related to quantization calibration
    "force_calibration": {
        "help": "Run quantization calibration even if the cache exists. (Only used for quantized models)",
        "action": "store_true",
    },
    "calib_batch_size": {
        "help": "Batch size for calibration.",
        "type": int,
        "default": 1,
    },
    "calib_max_batches": {
        "help": "Number of batches to run for calibration.",
        "type": int,
        "default": 500,
    },
    "cache_file": {
        "help": "Path to calibration cache.",
        "default": None,
    },
    "calib_data_map": {
        "help": "Path to the data map of the calibration set.",
        "default": None,
    },

    # Benchmark configuration arguments
    "scenario": {
        "help": "Name for the scenario. Used to generate engine name.",
    },
    "dla_core": {
        "help": "DLA core to use. Do not set if not using DLA",
        "default": None,
    },
    "model_path": {
        "help": "Path to the model (weights) file.",
    },
    "active_sms": {
        "help": "Control the percentage of active SMs while generating engines.",
        "type": int
    },

    # GNMT arguments
    "input_file": {
        "help": "Input file",
        "type": str,
    },
    "input_file_accuracy": {
        "help": "Input file for Accuracy run",
        "type": str,
    },
    "input_file_performance": {
        "help": "Input file for Performance run",
        "type": str,
    },
    "beam_size": {
        "help": "Beam size",
        "type": int,
    },
    "max_persistent_bs": {
        "help": "Maximum batch size to use for persistent encoder.",
        "type": int,
    },
    "homogeneous_cluster": {
        "help": "Assume every GPU in the cluster is similar.",
        "action": "store_true",
    },
    "enable_int8_generator": {
        "help": "Enable int8 Generator in GNMT.",
        "action": "store_true",
    },
    "seq_len_slots": {
        "help": "The number of TRT engines to be created for different sequence lengths",
        "type": int,
    },

    # Harness configuration arguments
    "log_dir": {
        "help": "Directory for all output logs.",
        "default": os.environ.get("LOG_DIR", "build/logs/default"),
    },
    "use_graphs": {
        "help": "Enable CUDA graphs",
    },

    # LWIS settings
    "devices": {
        "help": "Comma-separated list of numbered devices",
    },
    "map_path": {
        "help": "Path to map file for samples",
    },
    "tensor_path": {
        "help": "Path to preprocessed samples in .npy format",
    },
    "performance_sample_count": {
        "help": "Number of samples to load in performance set.  0=use default",
        "type": int,
    },
    "gpu_copy_streams": {
        "help": "Number of copy streams to use for GPU",
        "type": int,
    },
    "gpu_inference_streams": {
        "help": "Number of inference streams to use for GPU",
        "type": int,
    },
    "dla_copy_streams": {
        "help": "Number of copy streams to use for DLA",
        "type": int,
    },
    "dla_inference_streams": {
        "help": "Number of inference streams to use for DLA",
        "type": int,
    },
    "run_infer_on_copy_streams": {
        "help": "Run inference on copy streams.",
    },
    "warmup_duration": {
        "help": "Minimum duration to perform warmup for",
        "type": float,
    },
    "use_direct_host_access": {
        "help": "Use direct access to host memory for all devices",
    },
    "use_deque_limit": {
        "help": "Use a max number of elements dequed from work queue",
    },
    "deque_timeout_us": {
        "help": "Timeout in us for deque from work queue.",
        "type": int,
    },
    "use_batcher_thread_per_device": {
        "help": "Enable a separate batcher thread per device",
    },
    "use_cuda_thread_per_device": {
        "help": "Enable a separate cuda thread per device",
    },

    # Shared settings
    "mlperf_conf_path": {
        "help": "Path to mlperf.conf",
    },
    "user_conf_path": {
        "help": "Path to user.conf",
    },

    # Loadgen settings
    "test_mode": {
        "help": "Testing mode for Loadgen",
        "choices": ["SubmissionRun", "AccuracyOnly", "PerformanceOnly", "FindPeakPerformance"],
    },
    "min_duration_ms": {
        "help": "Minimum test duration",
        "type": int,
    },
    "max_duration_ms": {
        "help": "Maximum test duration",
        "type": int,
    },
    "min_query_count": {
        "help": "Minimum number of queries in test",
        "type": int,
    },
    "max_query_count": {
        "help": "Maximum number of queries in test",
        "type": int,
    },
    "qsl_rng_seed": {
        "help": "Seed for RNG that specifies which QSL samples are chosen for performance set and the order in which samples are processed in AccuracyOnly mode",
        "type": int,
    },
    "sample_index_rng_seed": {
        "help": "Seed for RNG that specifies order in which samples from performance set are included in queries",
        "type": int,
    },

    # Loadgen logging settings
    "logfile_suffix": {
        "help": "Specify the filename suffix for the LoadGen log files",
    },
    "logfile_prefix_with_datetime": {
        "help": "Prefix filenames for LoadGen log files",
        "action": "store_true",
    },
    "log_copy_detail_to_stdout": {
        "help": "Copy LoadGen detailed logging to stdout",
        "action": "store_true",
    },
    "disable_log_copy_summary_to_stdout": {
        "help": "Disable copy LoadGen summary logging to stdout",
        "action": "store_true",
    },
    "log_mode": {
        "help": "Logging mode for Loadgen",
        "choices": ["AsyncPoll", "EndOfTestOnly", "Synchronous"],
    },
    "log_mode_async_poll_interval_ms": {
        "help": "Specify the poll interval for asynchrounous logging",
        "type": int,
    },
    "log_enable_trace": {
        "help": "Enable trace logging",
    },

    # Server harness arguments
    "server_target_qps": {
        "help": "Target QPS for server scenario.",
        "type": int,
    },
    "server_target_latency_ns": {
        "help": "Desired latency constraint for server scenario",
        "type": int,
    },
    "server_target_latency_percentile": {
        "help": "Desired latency percentile constraint for server scenario",
        "type": float,
    },
    # not supported by current Loadgen - when support is added use the Loadgen default
    #"server_coalesce_queries": {
    #    "help": "Enable coalescing outstanding queries in the server scenario",
    #    "action": "store_true",
    #},
    "schedule_rng_seed": {
        "help": "Seed for RNG that affects the poisson arrival process in server scenario",
        "type": int,
    },
    "accuracy_log_rng_seed": {
        "help": "Affects which samples have their query returns logged to the accuracy log in performance mode.",
        "type": int,
    },

    # Single stream harness arguments
    "single_stream_expected_latency_ns": {
        "help": "Inverse of desired target QPS",
        "type": int,
    },
    "single_stream_target_latency_percentile": {
        "help": "Desired latency percentile for single stream scenario",
        "type": float,
    },

    # Offline harness arguments
    "offline_expected_qps": {
        "help": "Target samples per second rate for the SUT",
        "type": float,
    },

    # Multi stream harness arguments
    "multi_stream_target_qps": {
        "help": "Target QPS rate for the SUT",
        "type": float,
    },
    "multi_stream_target_latency_ns": {
        "help": "Desired latency constraint for multi stream scenario",
        "type": int,
    },
    "multi_stream_target_latency_percentile": {
        "help": "Desired latency percentile for multi stream scenario",
        "type": float,
    },
    "multi_stream_samples_per_query": {
        "help": "Expected samples per query for multi stream scenario",
        "type": int,
    },
    "multi_stream_max_async_queries": {
        "help": "Max number of asynchronous queries for multi stream scenario",
        "type": int,
    },

    # Args used by code.main
    "action": {
        "help": "generate_engines / run_harness / calibrate",
        "choices": ["generate_engines", "run_harness", "calibrate"],
    },
    "benchmarks": {
        "help": "Specify the benchmark(s) with a comma-separated list. " +
            "Choices: [\"resnet\", \"mobilenet\", \"ssd-large\", \"ssd-small\", \"gnmt\"] " +
            "Default: run all benchmarks.",
        "default": None,
    },
    "configs": {
        "help": "Specify the config files with a comma-separated list. " +
            "Wild card (*) is also allowed. If \"\", detect platform and attempt to load configs. " +
            "Default: \"\"",
        "default": "",
    },
    "scenarios": {
        "help": "Specify the scenarios with a comma-separated list. " +
            "Choices:[\"Server\", \"Offline\", \"SingleStream\", \"MultiStream\"] " +
            "Default: \"*\"",
        "default": None,
    },
    "no_gpu": {
        "help": "Do not perform action with GPU parameters (run on DLA only).",
        "action": "store_true",
    },
    "gpu_only": {
        "help": "Only perform action with GPU parameters (do not run DLA).",
        "action": "store_true",
    },

    # Args used for engine runners
    "engine_file": {
        "help": "File to load engine from",
    },
    "num_images": {
        "help": "Number of images to use for accuracy runner",
        "type": int,
    },
}

# ================== Argument groups ================== #

# Engine generation
PRECISION_ARGS = ["precision", "explicit_precision", "input_dtype", "input_format"]
CALIBRATION_ARGS = ["force_calibration", "calib_batch_size", "calib_max_batches", "cache_file",
    "calib_data_map", "model_path"]
GENERATE_ENGINE_ARGS = ["verbose", "dla_core", "gpu_batch_size", "dla_batch_size"] + PRECISION_ARGS + CALIBRATION_ARGS

# Harness framework arguments
LOADGEN_ARGS = ["test_mode", "min_duration_ms", "max_duration_ms", "min_query_count",
    "max_query_count", "qsl_rng_seed", "sample_index_rng_seed", "schedule_rng_seed", "accuracy_log_rng_seed", "logfile_suffix",
    "logfile_prefix_with_datetime", "log_copy_detail_to_stdout", "disable_log_copy_summary_to_stdout",
    "log_mode", "log_mode_async_poll_interval_ms", "log_enable_trace",
    "single_stream_target_latency_percentile", "multi_stream_target_latency_percentile",
    "multi_stream_target_qps", "multi_stream_target_latency_ns", "multi_stream_max_async_queries",
    "server_target_latency_percentile", "server_target_qps", "server_target_latency_ns" ]
LWIS_ARGS = ["devices", "gpu_copy_streams", "gpu_inference_streams",
    "dla_batch_size", "dla_copy_streams", "dla_inference_streams", "run_infer_on_copy_streams", "warmup_duration", "use_direct_host_access", "use_deque_limit", "deque_timeout_us",
    "use_batcher_thread_per_device", "use_cuda_thread_per_device"]
SHARED_ARGS = [ "use_graphs", "gpu_batch_size", "map_path", "tensor_path", "performance_sample_count", "mlperf_conf_path", "user_conf_path" ]
OTHER_HARNESS_ARGS = ["log_dir"]

HARNESS_ARGS = ["verbose", "scenario", "precision"] + LOADGEN_ARGS + LWIS_ARGS + SHARED_ARGS + OTHER_HARNESS_ARGS

# Scenario dependent arguments. These are prefixed with device: "gpu_", "dla_", "concurrent_"
OFFLINE_PARAMS = [ "offline_expected_qps" ]
SINGLE_STREAM_PARAMS = [ "single_stream_expected_latency_ns" ]
MULTI_STREAM_PARAMS = [ "multi_stream_samples_per_query" ]
SERVER_PARAMS = []

# Wrapper for scenario+harness
OFFLINE_HARNESS_ARGS = OFFLINE_PARAMS + HARNESS_ARGS
SINGLE_STREAM_HARNESS_ARGS = SINGLE_STREAM_PARAMS + HARNESS_ARGS
MULTI_STREAM_HARNESS_ARGS = MULTI_STREAM_PARAMS + HARNESS_ARGS
SERVER_HARNESS_ARGS = SERVER_PARAMS + HARNESS_ARGS #LOADGEN_ARGS + OTHER_HARNESS_ARGS + ["gpu_batch_size", "map_path", "tensor_path"]

# GNMT-specific arguments
GNMT_ENGINE_ARGS = [ "verbose", "gpu_batch_size", "precision", "beam_size", "max_persistent_bs", "enable_int8_generator", "seq_len_slots" ]
GNMT_HARNESS_ARGS = LOADGEN_ARGS + ["log_dir", "input_file", "input_file_accuracy", "input_file_performance", "use_graphs", "gpu_batch_size",
        "homogeneous_cluster", "scenario"]
GNMT_SERVER_ARGS = LOADGEN_ARGS + SERVER_PARAMS + ["input_file", "input_file_accuracy", "input_file_performance"]

# For code.main
MAIN_ARGS = ["action", "benchmarks", "configs", "scenarios", "no_gpu", "gpu_only"]

# For accuracy runners
ACCURACY_ARGS = ["verbose", "engine_file", "batch_size", "num_images"]

def parse_args(whitelist):
    parser = argparse.ArgumentParser()
    for flag in whitelist:
        if flag not in arguments_dict:
            raise IndexError("Unknown flag '{:}'".format(flag))

        parser.add_argument("--{:}".format(flag), **arguments_dict[flag])
    return vars(parser.parse_known_args()[0])
