# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

import json
import re

import os, sys
sys.path.insert(0, os.getcwd())

from code.common import logging
from code.common import args_to_string, load_configs, run_command
import code.common.arguments as common_args
from importlib import import_module

def get_benchmark(benchmark_name, conf):
    GNMTBuilder = import_module("code.gnmt.tensorrt.GNMT").GNMTBuilder
    return GNMTBuilder(conf)

def apply_overrides(config, keys):
    # Make a copy so we don't modify original dict
    config = dict(config)
    override_args = common_args.parse_args(keys)
    for key in override_args:
        # Unset values (None) and unset store_true values (False) are both false-y
        if override_args[key]:
            config[key] = override_args[key]
    return config

def handle_generate_engine(benchmark_name, config, gpu=True, dla=True):
    logging.info("Building engines for {:} benchmark in {:} scenario...".format(benchmark_name, config["scenario"]))

    arglist = common_args.GNMT_ENGINE_ARGS
    config = apply_overrides(config, arglist)

    config["batch_size"] = config["gpu_batch_size"]
    config["dla_core"] = None
    logging.info("Building GPU engine for {:}_{:}_{:}".format(config["system_id"], benchmark_name, config["scenario"]))
    b = get_benchmark(benchmark_name, config)
    b.build_engines()

    logging.info("Finished building engines for {:} benchmark in {:} scenario.".format(benchmark_name, config["scenario"]))

def handle_run_harness(benchmark_name, config):
    logging.info("Running harness for {:} benchmark in {:} scenario...".format(benchmark_name, config["scenario"]))

    arglist = common_args.GNMT_HARNESS_ARGS

    config = apply_overrides(config, arglist)

    config["dla_batch_size"] = None

    from code.gnmt.tensorrt.harness import GNMTHarness
    harness = GNMTHarness(config, name=benchmark_name)

    result = harness.run_harness()
    logging.info("Result: {:}".format(result))

    # Append result to perf result summary log.
    log_dir = config["log_dir"]
    summary_file = os.path.join(log_dir, "perf_harness_summary.json")
    results = {}
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            results = json.load(f)

    config_name = "{:}-{:}".format(config["system_id"], config["scenario"])
    if config_name not in results:
        results[config_name] = {}
    results[config_name][benchmark_name] = result

    with open(summary_file, "w") as f:
        json.dump(results, f)

    # Check accuracy from loadgen logs.
    accuracy = check_accuracy(os.path.join(log_dir, config["system_id"], benchmark_name, config["scenario"], "mlperf_log_accuracy.json"),
        benchmark_name, config)

    summary_file = os.path.join(log_dir, "accuracy_summary.json")
    results = {}
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            results = json.load(f)

    config_name = "{:}-{:}".format(config["system_id"], config["scenario"])
    if config_name not in results:
        results[config_name] = {}
    results[config_name][benchmark_name] = accuracy

    with open(summary_file, "w") as f:
        json.dump(results, f)

def check_accuracy(log_file, benchmark_name, config):

    accuracy_targets = {
        "gnmt": 23.9
    }
    threshold_ratios = {
        "gnmt": 0.99
    }

    if not os.path.exists(log_file):
        return "Cannot find accuracy JSON file."
    with open(log_file, "r") as f:
        loadgen_dump = json.load(f)
    if len(loadgen_dump) == 0:
        return "No accuracy results in PerformanceOnly mode."

    threshold = accuracy_targets[benchmark_name] * threshold_ratios[benchmark_name]
    if benchmark_name == "gnmt":
        cmd = "python3 build/inference/v0.5/translation/gnmt/tensorflow/process_accuracy.py --accuracy_log {:} \
            --reference build/preprocessed_data/nmt/GNMT/newstest2014.tok.bpe.32000.de".format(log_file)
        regex = r"BLEU: ([0-9\.]+)"
    else:
        raise ValueError("Unknown benchmark: {:}".format(benchmark_name))

    output = run_command(cmd, get_output=True)
    result_regex = re.compile(regex)
    accuracy = None
    with open(os.path.join(os.path.dirname(log_file), "accuracy.txt"), "w") as f:
        for line in output:
            print(line, file=f)

    for line in output:
        result_match = result_regex.match(line)
        if not result_match is None:
            accuracy = float(result_match.group(1))
            break

    accuracy_result = "PASSED" if accuracy is not None and accuracy >= threshold else "FAILED"

    if accuracy_result == "FAILED":
        raise RuntimeError("Accuracy = {:.3f}, Threshold = {:.3f}. Accuracy test {:}!".format(accuracy, threshold, accuracy_result))

    return "Accuracy = {:.3f}, Threshold = {:.3f}. Accuracy test {:}.".format(accuracy, threshold, accuracy_result)

def handle_calibrate(benchmark_name, config):
    logging.info("Generating calibration cache for Benchmark \"{:}\"".format(benchmark_name))
    config = apply_overrides(config, common_args.CALIBRATION_ARGS)
    config["dla_core"] = None
    b = get_benchmark(benchmark_name, config)
    b.calibrate()

def main():
    main_args = common_args.parse_args(common_args.MAIN_ARGS)

    scenarios = ["SingleStream", "Offline"]
    scenarios_legacy_map = {
        "single_stream": "SingleStream",
        "offline": "Offline"
    }
    if main_args["scenarios"] is not None:
        scenarios = main_args["scenarios"].split(",")
        for i, scenario in enumerate(scenarios):
            if scenario in scenarios_legacy_map:
                scenarios[i] = scenarios_legacy_map[scenario]

    config_files = ",".join(["measurements/Xavier/gnmt/{:}/config.json".format(i) for i in scenarios])

    logging.info("Using config files: {:}".format(str(config_files)))
    configs = load_configs(config_files)

    for config in configs:
        logging.info("Processing config \"{:}\"".format(config["config_name"]))

        benchmark_name = config["benchmark"]
        benchmark_conf = config[benchmark_name]

        # Passthrough for top level values
        benchmark_conf["system_id"] = config["system_id"]
        benchmark_conf["scenario"] = config["scenario"]
        benchmark_conf["benchmark"] = config["benchmark"]
        benchmark_conf["config_name"] = config["config_name"]

        if main_args["action"] == "generate_engines":
            handle_generate_engine(benchmark_name, benchmark_conf)
        elif main_args["action"] == "run_harness":
            handle_run_harness(benchmark_name, benchmark_conf)
        elif main_args["action"] == "calibrate":
            # To generate calibration cache, we only need to run each benchmark once. Use offline config.
            if benchmark_conf["scenario"] == "Offline":
                handle_calibrate(benchmark_name, benchmark_conf)

if __name__ == "__main__":
    main()
