#! /usr/bin/env python3
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

import argparse
import json
import os, sys
sys.path.insert(0, os.getcwd())

import code.common.arguments as common_args

def main():
    log_dir = common_args.parse_args(["log_dir"])["log_dir"]

    summary_file = os.path.join(log_dir, "perf_harness_summary.json")
    with open(summary_file) as f:
        results = json.load(f)

    print("")
    print("======================= Perf harness results: =======================")
    print("")

    for config_name in results:
        print("{:}:".format(config_name))
        for benchmark in results[config_name]:
            print("    {:}: {:}".format(benchmark, results[config_name][benchmark]))
        print("")

    summary_file = os.path.join(log_dir, "accuracy_summary.json")
    with open(summary_file) as f:
        results = json.load(f)

    print("")
    print("======================= Accuracy results: =======================")
    print("")

    for config_name in results:
        print("{:}:".format(config_name))
        for benchmark in results[config_name]:
            print("    {:}: {:}".format(benchmark, results[config_name][benchmark]))
        print("")

if __name__=="__main__":
    main()
