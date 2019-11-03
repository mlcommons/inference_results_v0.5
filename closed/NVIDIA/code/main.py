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

from code.common.scopedMPS import ScopedMPS, turn_off_mps
from code.common import logging
from code.common import args_to_string, find_config_files, load_configs, run_command
import code.common.arguments as common_args
from importlib import import_module
import multiprocessing as mp
from multiprocessing import Process

def get_benchmark(benchmark_name, conf):
    # Do not use a map. We want to import benchmarks as we need them, because some take
    # time to load due to plugins.
    if benchmark_name == "resnet":
        ResNet50 = import_module("code.resnet.tensorrt.ResNet50").ResNet50
        return ResNet50(conf)
    elif benchmark_name == "mobilenet":
        MobileNet = import_module("code.mobilenet.tensorrt.MobileNet").MobileNet
        return MobileNet(conf)
    elif benchmark_name == "ssd-small":
        SSDMobileNet = import_module("code.ssd-small.tensorrt.SSDMobileNet").SSDMobileNet
        return SSDMobileNet(conf)
    elif benchmark_name == "ssd-large":
        SSDResNet34 = import_module("code.ssd-large.tensorrt.SSDResNet34").SSDResNet34
        return SSDResNet34(conf)
    elif benchmark_name == "gnmt":
        GNMTBuilder = import_module("code.gnmt.tensorrt.GNMT").GNMTBuilder
        return GNMTBuilder(conf)
    else:
        raise ValueError("Unknown benchmark: {:}".format(benchmark_name))

def apply_overrides(config, keys):
    # Make a copy so we don't modify original dict
    config = dict(config)
    override_args = common_args.parse_args(keys)
    for key in override_args:
        # Unset values (None) and unset store_true values (False) are both false-y
        if override_args[key]:
            config[key] = override_args[key]
    return config

def launch_handle_generate_engine(benchmark_name, config, gpu, dla):
    retries = 3
    timeout = 7200
    success = False
    for i in range(retries):
        # Build engines in another process to make sure we exit with clean cuda context so that MPS can be turned off.
        from code.main import handle_generate_engine
        p = Process(target=handle_generate_engine, args=(benchmark_name, config, gpu, dla))
        p.start()
        try:
            p.join(timeout)
        except KeyboardInterrupt:
            p.terminate()
            p.join(timeout)
            raise KeyboardInterrupt
        if p.exitcode == 0:
            success = True
            break

    if not success:
        raise RuntimeError("Building engines failed!")

def handle_generate_engine(benchmark_name, config, gpu=True, dla=True):
    logging.info("Building engines for {:} benchmark in {:} scenario...".format(benchmark_name, config["scenario"]))

    if benchmark_name == "gnmt":
        arglist = common_args.GNMT_ENGINE_ARGS
    else:
        arglist = common_args.GENERATE_ENGINE_ARGS
    config = apply_overrides(config, arglist)

    if dla and "dla_batch_size" in config:
        config["batch_size"] = config["dla_batch_size"]
        logging.info("Building DLA engine for {:}_{:}_{:}".format(config["system_id"], benchmark_name, config["scenario"]))
        b = get_benchmark(benchmark_name, config)
        b.build_engines()

    if gpu and "gpu_batch_size" in config:
        config["batch_size"] = config["gpu_batch_size"]
        config["dla_core"] = None
        logging.info("Building GPU engine for {:}_{:}_{:}".format(config["system_id"], benchmark_name, config["scenario"]))
        b = get_benchmark(benchmark_name, config)
        b.build_engines()

    if gpu and config["scenario"] == "Server" and benchmark_name == "gnmt":
        b = get_benchmark(benchmark_name, config)
        b.build_engines()

    logging.info("Finished building engines for {:} benchmark in {:} scenario.".format(benchmark_name, config["scenario"]))

def handle_run_harness(benchmark_name, config, gpu=True, dla=True):
    logging.info("Running harness for {:} benchmark in {:} scenario...".format(benchmark_name, config["scenario"]))

    if config["scenario"] == "SingleStream":
        arglist = common_args.SINGLE_STREAM_HARNESS_ARGS
    elif config["scenario"] == "Offline":
        arglist = common_args.OFFLINE_HARNESS_ARGS
    elif config["scenario"] == "MultiStream":
        arglist = common_args.MULTI_STREAM_HARNESS_ARGS
    elif config["scenario"] == "Server":
        arglist = common_args.SERVER_HARNESS_ARGS

    if benchmark_name == "gnmt":
        arglist = common_args.GNMT_HARNESS_ARGS

    config = apply_overrides(config, arglist)

    # Validate arguments

    if not dla:
        config["dla_batch_size"] = None
    if not gpu:
        config["gpu_batch_size"] = None

    if benchmark_name == "gnmt":
        from code.common.harness import GNMTHarness
        harness = GNMTHarness(config, name=benchmark_name)
    else:
        from code.common.harness import BenchmarkHarness
        harness = BenchmarkHarness(config, name=benchmark_name)

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
        "resnet": 76.46,
        "mobilenet": 71.68,
        "ssd-large": 20.0,
        "ssd-small": 22.0,
        "gnmt": 23.9
    }
    threshold_ratios = {
        "resnet": 0.99,
        "mobilenet": 0.98,
        "ssd-large": 0.99,
        "ssd-small": 0.99,
        "gnmt": 0.99
    }

    if not os.path.exists(log_file):
        return "Cannot find accuracy JSON file."
    with open(log_file, "r") as f:
        loadgen_dump = json.load(f)
    if len(loadgen_dump) == 0:
        return "No accuracy results in PerformanceOnly mode."

    threshold = accuracy_targets[benchmark_name] * threshold_ratios[benchmark_name]
    if benchmark_name in ["resnet", "mobilenet"]:
        cmd = "python3 build/inference/v0.5/classification_and_detection/tools/accuracy-imagenet.py --mlperf-accuracy-file {:} \
            --imagenet-val-file data_maps/imagenet/val_map.txt --dtype int32 ".format(log_file)
        regex = r"accuracy=([0-9\.]+)%, good=[0-9]+, total=[0-9]+"
    elif benchmark_name == "ssd-small":
        cmd = "python3 build/inference/v0.5/classification_and_detection/tools/accuracy-coco.py --mlperf-accuracy-file {:} \
            --coco-dir {:} --output-file build/ssd-small-results.json".format(
            log_file, os.path.join(os.environ.get("PREPROCESSED_DATA_DIR", "build/preprocessed_data"), "coco"))
        regex = r"mAP=([0-9\.]+)%"
    elif benchmark_name == "ssd-large":
        cmd = "python3 build/inference/v0.5/classification_and_detection/tools/accuracy-coco.py --mlperf-accuracy-file {:} \
            --coco-dir {:} --output-file build/ssd-large-results.json --use-inv-map".format(
            log_file, os.path.join(os.environ.get("PREPROCESSED_DATA_DIR", "build/preprocessed_data"), "coco"))
        regex = r"mAP=([0-9\.]+)%"
    elif benchmark_name == "gnmt":
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
    # Turn off MPS in case it's turned on.
    turn_off_mps()

    main_args = common_args.parse_args(common_args.MAIN_ARGS)

    benchmarks = ["mobilenet", "resnet", "ssd-small", "ssd-large", "gnmt"]
    benchmarks_legacy_map = {
        "ResNet50": "resnet",
        "MobileNet": "mobilenet",
        "SSDMobileNet": "ssd-small",
        "SSDResNet34": "ssd-large",
        "GNMT": "gnmt"
    }
    if main_args["benchmarks"] is not None:
        benchmarks = main_args["benchmarks"].split(",")
        for i, benchmark in enumerate(benchmarks):
            if benchmark in benchmarks_legacy_map:
                benchmarks[i] = benchmarks_legacy_map[benchmark]

    scenarios = ["SingleStream", "MultiStream", "Offline", "Server"]
    scenarios_legacy_map = {
        "single_stream": "SingleStream",
        "multi_stream": "MultiStream",
        "offline": "Offline",
        "server": "Server"
    }
    if main_args["scenarios"] is not None:
        scenarios = main_args["scenarios"].split(",")
        for i, scenario in enumerate(scenarios):
            if scenario in scenarios_legacy_map:
                scenarios[i] = scenarios_legacy_map[scenario]

    # Automatically detect architecture and scenarios and load configs
    config_files = main_args["configs"]
    if config_files == "":
        config_files = find_config_files(benchmarks, scenarios)
        if config_files == "":
            logging.warn("Cannot find any valid configs for the specified benchmarks scenarios.")
            return

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

        need_gpu = not main_args["no_gpu"]
        need_dla = not main_args["gpu_only"]

        if main_args["action"] == "generate_engines":
            # Turn on MPS if server scenario and if active_sms is specified.
            benchmark_conf = apply_overrides(benchmark_conf, ["active_sms"])
            active_sms = benchmark_conf.get("active_sms", None)
            if config["scenario"] == "Server" and active_sms is not None and active_sms < 100:
                with ScopedMPS(active_sms):
                    launch_handle_generate_engine(benchmark_name, benchmark_conf, need_gpu, need_dla)
            else:
                launch_handle_generate_engine(benchmark_name, benchmark_conf, need_gpu, need_dla)
        elif main_args["action"] == "run_harness":
            handle_run_harness(benchmark_name, benchmark_conf, need_gpu, need_dla)
        elif main_args["action"] == "calibrate":
            # To generate calibration cache, we only need to run each benchmark once. Use offline config.
            if benchmark_conf["scenario"] == "Offline":
                handle_calibrate(benchmark_name, benchmark_conf)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
