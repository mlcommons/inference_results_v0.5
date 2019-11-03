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
from code.common import logging, dict_get

import tensorrt as trt

class AbstractBuilder(object):
    def __init__(self):
        raise NotImplementedError("AbstractBuilder cannot be called directly")

    def build_engines(self):
        raise NotImplementedError("AbstractBuilder cannot be called directly")

    def calibrate(self):
        raise NotImplementedError("AbstractBuilder cannot be called directly")

class BenchmarkBuilder(AbstractBuilder):

    """
    Constructor
    :param args: arguments represented by a dictionary
    :param name: name of the benchmark
    """
    def __init__(self, args, name="", workspace_size=(1 << 30)):
        self.name = name
        self.args = args

        # Configuration variables
        self.verbose = dict_get(args, "verbose", default=False)
        if self.verbose:
            logging.info("========= BenchmarkBuilder Arguments =========")
            for arg in args:
                logging.info("{:}={:}".format(arg, args[arg]))

        self.system_id = args["system_id"]
        self.scenario = args["scenario"]
        self.engine_dir = "./build/engines/{:}/{:}/{:}".format(self.system_id, self.name, self.scenario)

        # Set up logger, builder, and network.
        self.logger = trt.Logger(trt.Logger.VERBOSE if self.verbose else trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.builder = trt.Builder(self.logger)
        self.builder_config = self.builder.create_builder_config()
        self.builder_config.max_workspace_size = workspace_size

        # Precision variables
        self.input_dtype = dict_get(args, "input_dtype", default="fp32")
        self.input_format = dict_get(args, "input_format", default="linear")
        self.precision = dict_get(args, "precision", default="int8")
        self.explicit_precision = dict_get(args, "explicit_precision", default=False)
        if self.precision == "fp16":
            self.apply_flag(trt.BuilderFlag.FP16)
        elif self.precision == "int8":
            self.apply_flag(trt.BuilderFlag.INT8)

        # Device variables
        self.device_type = "gpu"
        self.dla_core = args.get("dla_core", None)
        if self.dla_core is not None:
            logging.info("Using DLA: Core {:}".format(self.dla_core))
            self.device_type = "dla"
            self.apply_flag(trt.BuilderFlag.GPU_FALLBACK)
            self.builder_config.default_device_type = trt.DeviceType.DLA
            self.builder_config.DLA_core = int(self.dla_core)

        self.initialized = False

    """
    Builds the network in preparation for building the engine. This method must be implemented by
    the subclass.

    The implementation should also set self.initialized to True.
    """
    def initialize(self):
        raise NotImplementedError("BenchmarkBuilder.initialize() should build the network")

    """
    Apply a TRT builder flag.
    """
    def apply_flag(self, flag):
        self.builder_config.flags = (self.builder_config.flags) | (1 << int(flag))

    """
    Calls self.initialize() if it has not been called yet. Builds and saves the engine.
    """
    def build_engines(self):
        if not self.initialized:
            self.initialize()

        if self.scenario == "SingleStream":
            batch_size = 1
        elif self.scenario in ["Server", "Offline", "MultiStream"]:
            batch_size = self.args.get("batch_size", 1)
        else:
            raise ValueError("Invalid scenario: {:}".format(self.scenario))

        # Create output directory if it does not exist.
        if not os.path.exists(self.engine_dir):
            os.makedirs(self.engine_dir)

        for bs in self.args.get("batch_sizes", [batch_size]):
            engine_name = "{:}/{:}-{:}-{:}-b{:}-{:}.plan".format(
                self.engine_dir, self.name, self.scenario,
                self.device_type, bs, self.precision)
            logging.info("Building {:}".format(engine_name))

            # Build engines
            self.builder.max_batch_size = bs
            engine = self.builder.build_engine(self.network, self.builder_config)
            buf = engine.serialize()
            with open(engine_name, 'wb') as f:
                f.write(buf)

    """
    Generate a new calibration cache.
    """
    def calibrate(self):
        self.need_calibration = True
        self.calibrator.clear_cache()
        self.initialize()
        # Generate a dummy engine to generate a new calibration cache.
        self.builder.max_batch_size = 1
        engine = self.builder.build_engine(self.network, self.builder_config)
