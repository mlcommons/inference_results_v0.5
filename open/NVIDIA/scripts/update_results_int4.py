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

import os, sys
sys.path.insert(0, os.getcwd())

import re
import sys
import shutil
import glob
import argparse

system_list = ["T4x8", "T4x20", "TitanRTXx4"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", "-d",
        help="Specifies the directory containing the logs.",
        default="build/logs"
    )
    args = parser.parse_args()

    for system_id in system_list:
        for benchmark in ["resnet"]:
            for scenario in ["Offline"]:
                print(">>>>>>>> Processing {:}-{:}-{:} <<<<<<<<".format(system_id, benchmark, scenario))

                accu_glob = os.path.join(args.input_dir, "*", system_id, benchmark, scenario, "accuracy", "mlperf_log_accuracy.json")
                accu_list = glob.glob(accu_glob)

                perf_glob = os.path.join(args.input_dir, "*", system_id, benchmark, scenario, "performance", "run_1", "mlperf_log_accuracy.json")
                input_list = glob.glob(perf_glob)

                perf_list = []
                for input_file in input_list:
                    summary = input_file.replace("_accuracy.json", "_summary.txt")
                    with open(summary) as f:
                        for line in f:
                            match = re.match(r"Result is : (VALID|INVALID)", line)
                            if match is not None and match.group(1) == "VALID":
                                perf_list.append(input_file)
                                break

                #### Update accuracy run
                if len(accu_list) == 0:
                    print("WARNING: Cannot find valid accuracy run. Do you want to proceed? (Enter to proceed, Ctrl+C to abort)")
                    input()
                else:
                    if len(accu_list) > 1:
                        print("WARNING: Found {:d} accuracy runs, which is more than needed. Empirically choose the last one.".format(len(accu_list)))
                        print(accu_list)
                    output_dir = os.path.join("results", system_id, benchmark, scenario, "accuracy")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    for suffix in ["_accuracy.json", "_detail.txt", "_summary.txt"]:
                        input_file = accu_list[-1].replace("_accuracy.json", suffix)
                        output_file = os.path.join(output_dir, "mlperf_log{:}".format(suffix))
                        print("Copy {:} -> {:}".format(input_file, output_file))
                        shutil.copy(input_file, output_file)

                    input_file = os.path.join(os.path.dirname(input_file), "accuracy.txt")
                    output_file = os.path.join(output_dir, "accuracy.txt")
                    print("Copy {:} -> {:}".format(input_file, output_file))
                    shutil.copy(input_file, output_file)

                #### Update perf run
                perf_count = 1 if scenario != "Server" else 5
                if len(perf_list) < perf_count:
                    print("WARNING: Cannot find enough passing perf runs. Only found {:d} runs. Do you want to proceed? (Enter to proceed, Ctrl+C to abort)".format(len(perf_list)))
                    input()
                elif len(perf_list) > perf_count:
                    print("WARNING: Found {:d} passing perf runs, which is more than needed. Empirically choose the last passing one(s).".format(len(perf_list)))
                    print(perf_list)
                    perf_list = perf_list[-perf_count:]

                for run_idx in range(0, len(perf_list)):
                    output_dir = os.path.join("results", system_id, benchmark, scenario, "performance", "run_{:d}".format(run_idx+1))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    for suffix in ["_accuracy.json", "_detail.txt", "_summary.txt"]:
                        input_file = perf_list[run_idx].replace("_accuracy.json", suffix)
                        output_file = os.path.join(output_dir, "mlperf_log{:}".format(suffix))
                        print("Copy {:} -> {:}".format(input_file, output_file))
                        shutil.copy(input_file, output_file)

    print("Done!")

if __name__ == '__main__':
    main()
