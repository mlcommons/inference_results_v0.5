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

import re
import os, sys
sys.path.insert(0, os.getcwd())
from code.common import logging, dict_get, run_command, args_to_string
import code.common.arguments as common_args

plugin_map = {
    "ssd-large": ["build/plugins/NMSOptPlugin/libnmsoptplugin.so"],
    "ssd-small": ["build/plugins/NMSOptPlugin/libnmsoptplugin.so"],
}

scenario_result_regex = {
    "SingleStream": r"([0-9]+th percentile latency \(ns\) +: [0-9\.]+)",
    "MultiStream": r"(Samples per query : [0-9\.]+)",
    "Offline": r"(Samples per second: [0-9\.]+)",
    "Server": r"(Scheduled samples per second +: [0-9\.]+)",
}

benchmark_qsl_size_map = {
    # See: https://github.com/mlperf/inference_policies/blob/master/inference_rules.adoc#benchmarks-1
    "resnet": 1024,
    "mobilenet": 1024,
    "ssd-large": 64,
    "ssd-small": 256,
}

class BenchmarkHarness():

    def __init__(self, args, name=""):
        print (args)
        self.args = args
        self.name = name
        self.verbose = dict_get(args, "verbose", default=None)
        if self.verbose:
            logging.info("===== Harness arguments for {:} =====".format(name))
            for key in args:
                logging.info("{:}={:}".format(key, args[key]))

        self.system_id = args["system_id"]
        self.scenario = args["scenario"]
        self.engine_dir = "./build/engines/{:}/{:}/{:}".format(self.system_id, self.name, self.scenario)
        self.precision = args["precision"]

        self.has_gpu = dict_get(args, "gpu_batch_size", default=None) is not None
        self.has_dla = dict_get(args, "dla_batch_size", default=None) is not None

        self.enumerate_engines()

    def enumerate_engines(self):
        if self.has_gpu:
            self.gpu_engine = self._get_engine_name("gpu", self.args["gpu_batch_size"])
            self.check_file_exists(self.gpu_engine)

        if self.has_dla:
            self.dla_engine = self._get_engine_name("dla", self.args["dla_batch_size"])
            self.check_file_exists(self.dla_engine)

    def _get_engine_name(self, device_type, batch_size):
        return "{:}/{:}-{:}-{:}-b{:}-{:}.plan".format(self.engine_dir, self.name, self.scenario,
                device_type, batch_size, self.precision)

    def build_default_flags(self, custom_args):
        flag_dict = {}
        flag_dict["verbose"] = self.verbose

        # Handle plugins
        if self.name in plugin_map:
            plugins = plugin_map[self.name]
            for plugin in plugins:
                self.check_file_exists(plugin)
            flag_dict["plugins"] = ",".join(plugins)

        # Generate flags for logfile names.
        log_dir = os.path.join(self.args["log_dir"], self.system_id, self.name, self.scenario)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        flag_dict["logfile_outdir"] = log_dir
        flag_dict["logfile_prefix"] = "mlperf_log_"

        # Handle custom arguments
        for arg in custom_args:
            val = dict_get(self.args, arg, None)
            if val is not None:
                flag_dict[arg] = val

        return flag_dict

    def build_configs(self, flag_dict):        
        # ideally, these values would be in mlperf.conf.  since they aren't, write them into user.conf using these values.
        # QSL Seed:      0x2b7e 1516 28ae d2a6
        # Schedule Seed: 0x3243 f6a8 885a 308d
        # Sample Seed:   0x093c 467e 37db 0c7a
        seeds_map = {
            "qsl_rng_seed": "3133965575612453542",
            "sample_index_rng_seed": "665484352860916858",
            "schedule_rng_seed": "3622009729038561421",
        }

        # required settings for each scenario
        required_settings_map = {
            "SingleStream": ["qsl_rng_seed", "sample_index_rng_seed", "schedule_rng_seed"], # "single_stream_expected_latency_ns", See: https://github.com/mlperf/inference/issues/471
            "Offline": ["offline_expected_qps", "qsl_rng_seed", "sample_index_rng_seed", "schedule_rng_seed"],
            "MultiStream": ["multi_stream_samples_per_query", "qsl_rng_seed", "sample_index_rng_seed", "schedule_rng_seed"],
            "Server": ["server_target_qps", "qsl_rng_seed", "sample_index_rng_seed", "schedule_rng_seed"],
        }

        # optional settings that we support overriding
        optional_settings_map = {
            "SingleStream": [ "single_stream_target_latency_percentile", "min_query_count" ],
            "Offline": [ "min_query_count" ],
            "MultiStream": [ "multi_stream_target_qps", "multi_stream_target_latency_ns", "multi_stream_max_async_queries", "multi_stream_target_latency_percentile", "min_query_count" ],
            "Server": [ "server_target_latency_percentile", "server_target_latency_ns", "min_query_count" ],
        }

        # option name to config file map
        options_map = {
            "single_stream_expected_latency_ns": "target_latency",
            "offline_expected_qps": "target_qps",
            "multi_stream_samples_per_query": "samples_per_query",
            "server_target_qps": "target_qps",
        }

        parameter_scaling_map = {
            "target_latency": 1 / 1000000.0,
            "target_latency_percentile": 100.0,
        }

        system = self.system_id
        benchmark = self.name
        scenario = self.scenario

        mlperf_conf_path = "measurements/{:}/{:}/{:}/mlperf.conf".format(system, benchmark, scenario)
        user_conf_path = "measurements/{:}/{:}/{:}/user.conf".format(system, benchmark, scenario)

        # setup paths
        if "mlperf_conf_path" not in flag_dict:
            flag_dict["mlperf_conf_path"] = mlperf_conf_path
        if "user_conf_path" not in flag_dict:
            flag_dict["user_conf_path"] = user_conf_path

        # assign seed values
        for name, value in seeds_map.items():
            if name not in flag_dict:
                flag_dict[name] = value

        # auto-generate user.conf
        with open(user_conf_path, "w") as f:
            for param in required_settings_map[scenario]:
                param_name = param
                if param in options_map:
                    param_name = options_map[param]
                value = flag_dict[param]
                if param_name in parameter_scaling_map:
                    value = value * parameter_scaling_map[param_name]
                f.write("*.{:}.{:} = {:}\n".format(scenario, param_name, value))
                flag_dict[param] = None
            for param in optional_settings_map[scenario]:
                if param not in flag_dict: continue
                param_name = param
                if param in options_map:
                    param_name = options_map[param]
                value = flag_dict[param]
                if param_name in parameter_scaling_map:
                    value = value * parameter_scaling_map[param_name]
                f.write("*.{:}.{:} = {:}\n".format(scenario, param_name, value))
                flag_dict[param] = None

    def run_harness(self):
        executable = "./build/bin/harness_default"
        self.check_file_exists(executable)

        # These arguments are in self.args, passed in via code.main, which handles override arguments.
        harness_args = common_args.LOADGEN_ARGS + common_args.LWIS_ARGS + common_args.SHARED_ARGS
        flag_dict = self.build_default_flags(harness_args)

        # Handle engines
        if self.has_gpu:
            flag_dict["gpu_engines"] = self.gpu_engine
        if self.has_dla:
            flag_dict["dla_engines"] = self.dla_engine

        # Handle performance sample count
        flag_dict["performance_sample_count"] = benchmark_qsl_size_map[self.name]

        # Handle the expected qps values
        if self.has_gpu and self.has_dla:
            prefix = "concurrent_"
        elif self.has_gpu:
            prefix = "gpu_"
            flag_dict["max_dlas"] = 0
        elif self.has_dla:
            prefix = "dla_"
            flag_dict["max_dlas"] = 1
        else:
            raise ValueError("Cannot specify --no_gpu and --gpu_only at the same time")

        if self.scenario == "SingleStream":
            harness_flags = common_args.SINGLE_STREAM_PARAMS
        elif self.scenario == "Offline":
            harness_flags = common_args.OFFLINE_PARAMS
        elif self.scenario == "MultiStream":
            harness_flags = common_args.MULTI_STREAM_PARAMS
        elif self.scenario == "Server":
            harness_flags = common_args.SERVER_PARAMS

            # use jemalloc2 for server scenario.
            executable = "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 " + executable
        else:
            raise ValueError("Invalid scenario: {:}".format(self.scenario))

        for arg in harness_flags:
            val = dict_get(self.args, prefix+arg, None)
            if val is None:
                raise ValueError("Missing required key {:}".format(prefix+arg))
            flag_dict[arg] = val

        # Handle configurations
        self.build_configs(flag_dict)

        argstr = args_to_string(flag_dict) + " --scenario " + self.scenario + " --model " + self.name

        if self.name in ["ssd-small", "ssd-large"]:
            argstr += " --response_postprocess coco"

        cmd = "{:} {:}".format(executable, argstr)
        output = run_command(cmd, get_output=True)

        # Return harness result.
        return self.harness_get_result(output, scenario_result_regex[self.scenario])

    def harness_get_result(self, output, regex):
        # All harness outputs should have a result string
        result_regex = re.compile(regex)
        result_string = ""

        # All but the server harness should have an output with a validity message
        valid_regex = re.compile(r"(Result is : (VALID|INVALID))")
        valid_string = ""

        # Iterate through the harness output
        for line in output:
            # Check for the result string
            result_match = result_regex.match(line)
            if not result_match is None:
                result_string = result_match.group(1)
                break

        for line in output:
            # Check for the validity string
            valid_match = valid_regex.match(line)
            if not valid_match is None:
                valid_string = valid_match.group(1)
                break

        if result_string == "":
            return "Cannot find performance result. Maybe you are running in AccuracyOnly mode."
        elif valid_string == "":
            return result_string + " but cannot find validity result."
        else:
            return result_string + " and " + valid_string

    def check_file_exists(self, f):
        if not os.path.isfile(f):
            raise RuntimeError("File {:} does not exist.".format(f))

class GNMTHarness(BenchmarkHarness):

    def __init__(self, args, name=""):
        super().__init__(args, name=name)

    def check_dir_exists(self, d):
        if not os.path.isdir(d):
            raise RuntimeError("Directory {:} does not exist.".format(d))

    def enumerate_engines(self):
        self.engines = []

        if self.scenario == "Server":
            batch_sizes = self.args["batch_sizes"]
        else:
            batch_sizes = [ self.args["gpu_batch_size"] ]

        for batch_size in batch_sizes:
            engine_name = self._get_engine_name("gpu", batch_size)
            self.check_dir_exists(engine_name)
            self.engines.append(engine_name)

    def run_harness(self):
        if self.scenario == "Server":
            executable = "./build/bin/harness_gnmt_server"
            harness_args = common_args.GNMT_SERVER_ARGS
        else:
            executable = "./build/bin/harness_gnmt_default"
            harness_args = common_args.GNMT_HARNESS_ARGS

        self.check_file_exists(executable)
        flag_dict = self.build_default_flags(harness_args)

        # Scenario based arguments
        if self.scenario == "Offline":
            scenario_args = common_args.OFFLINE_PARAMS
        elif self.scenario == "SingleStream":
            scenario_args = common_args.SINGLE_STREAM_PARAMS
        else:
            scenario_args = []

        for key in scenario_args:
            flag_dict[key] = dict_get(self.args, "gpu_"+key, None)

        engine_flag = "engine" if len(self.engines) == 1 else "engines"
        flag_dict[engine_flag] = ",".join(self.engines)

        # Remove the batch size flags
        flag_dict["batch_sizes"] = None
        flag_dict["gpu_batch_size"] = None

        # Choose input file based on test mode
        if ((flag_dict.get("test_mode", None) == "PerformanceOnly" or flag_dict.get("test_mode", None) is None)
            and flag_dict.get("input_file_performance", None) is not None):
            flag_dict["input_file"] = flag_dict["input_file_performance"]
        elif flag_dict.get("input_file_accuracy", None) is not None:
            flag_dict["input_file"] = flag_dict["input_file_accuracy"]
        flag_dict["input_file_performance"] = None
        flag_dict["input_file_accuracy"] = None

        # Handle configurations
        self.build_configs(flag_dict)

        argstr = args_to_string(flag_dict)
        cmd = "{:} {:}".format(executable, argstr)
        output = run_command(cmd, get_output=True)

        # Return harness result.
        return self.harness_get_result(output, scenario_result_regex[self.scenario])
