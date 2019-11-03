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
import shutil
sys.path.insert(0, os.getcwd())
from code.common import logging, dict_get, run_command

g_calibration_cache = "code/gnmt/tensorrt/data/Int8CalibrationCache"

class GNMTBuilder():

    def __init__(self, args):
        self.args = args
        self.name = "gnmt"
        self.system_id = args["system_id"]
        self.scenario = args["scenario"]
        self.engine_dir = "./build/engines/{:}/{:}/{:}".format(self.system_id, self.name, self.scenario)
        self.device_type = "gpu"

        self.precision = args["precision"]
        self.precision_flag = ""
        if (self.precision == "int8") or (self.precision == "fp16"):
            self.precision_flag = "-t fp16"
        if dict_get(args, "enable_int8_generator", default=False):
            self.precision_flag += " --int8Generator --calibration_cache {}".format(g_calibration_cache)

    def build_engines(self):
        if self.scenario == "MultiStream":
            raise NotImplementedError("GNMT MultiStream scenario is not yet implemented")
        elif self.scenario == "SingleStream":
            batch_sizes = [ self.args["batch_size"] ]
        elif self.scenario == "Offline":
            batch_sizes = [ self.args["batch_size"] ]
        elif self.scenario == "Server":
            batch_sizes = self.args["batch_sizes"]
        else:
            raise ValueError("Invalid scenario: {:}".format(self.scenario))

        beam_size = dict_get(self.args, "beam_size", default=10)
        seq_len_slots = dict_get(self.args, "seq_len_slots", default=1)
        mpbs_flag = ""
        if "max_persistent_bs" in self.args:
            mpbs_flag = "--max_persistent_bs {:}".format(self.args["max_persistent_bs"])

        if not os.path.exists(self.engine_dir):
            os.makedirs(self.engine_dir)

        for batch_size in batch_sizes:
            engine_name = "{:}/{:}-{:}-gpu-b{:}-{:}.plan".format(self.engine_dir, self.name, self.scenario, batch_size, self.precision)
            logging.info("Building {:}".format(engine_name))


            cmd = "build/bin/GNMT/gnmt --seq_len_slots {seq_len_slots} --bm {beam_size} --build_only --bs {batch_size} --store_engine {engine_name} {max_persistent_bs} {precision}".format(
                seq_len_slots=seq_len_slots,
                beam_size=beam_size,
                batch_size=batch_size,
                engine_name=engine_name,
                max_persistent_bs=mpbs_flag,
                precision=self.precision_flag,
            )
            run_command(cmd)

    def calibrate(self):
        beam_size = 1
        batch_size = 1
        num_batches = 64
        output_dir_1 = "calib_phase_1"
        output_dir_2 = "calib_phase_2"

        try:
            phase_1 = "./build/bin/GNMT/gnmt --calibration_phase 1 --input_file {input} --num_batches {num_batches} --bm {beam_size} --bs {batch_size} --output_dir {output_dir}".format(
                    input="build/inference/calibration/translation/calibration_data.tok.bpe.en",
                    num_batches = num_batches,
                    beam_size = beam_size,
                    batch_size = batch_size,
                    output_dir = output_dir_1
                    )

            run_command(phase_1)

            # Get the directory to raw output
            calib_data = None
            for d in os.listdir(output_dir_1):
                if "gnmt_tensors" in d:
                    calib_data = os.path.join(output_dir_1, d)
            assert(calib_data != None)

            phase_2 = "./build/bin/GNMT/gnmt --calibration_phase 2 --bm {beam_size} --calibration_data {input}  --calibration_cache {calib_cache} --output_dir {output_dir}".format(
                    input = calib_data,
                    beam_size = beam_size,
                    batch_size = batch_size,
                    calib_cache = g_calibration_cache,
                    output_dir = output_dir_2
                    )

            run_command(phase_2)

        # Remove the generated intermediate output files
        finally:
            if os.path.exists(output_dir_1):
                shutil.rmtree(output_dir_1)
            if os.path.exists(output_dir_2):
                shutil.rmtree(output_dir_2)
