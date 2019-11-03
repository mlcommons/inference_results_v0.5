#!/usr/bin/env python3
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

import tensorrt as trt
import os, sys
sys.path.insert(0, os.getcwd())

from importlib import import_module
from code.common import logging, dict_get
from code.common.builder import BenchmarkBuilder
RN50Calibrator = import_module("code.resnet.tensorrt.calibrator").RN50Calibrator
parse_calibration = import_module("code.resnet.tensorrt.res2_fusions").parse_calibration
fuse_br1_br2c_onnx = import_module("code.resnet.tensorrt.res2_fusions").fuse_br1_br2c_onnx
fuse_br2b_br2c_onnx = import_module("code.resnet.tensorrt.res2_fusions").fuse_br2b_br2c_onnx

class ResNet50(BenchmarkBuilder):

    def __init__(self, args):
        super().__init__(args, name="resnet")

        # Model path
        self.model_path = dict_get(args, "model_path", default="build/models/ResNet50/resnet50_v1.onnx")

        if self.precision == "int8":
            # Get calibrator variables
            calib_batch_size = dict_get(self.args, "calib_batch_size", default=1)
            calib_max_batches = dict_get(self.args, "calib_max_batches", default=500)
            force_calibration = dict_get(self.args, "force_calibration", default=False)
            cache_file = dict_get(self.args, "cache_file", default="code/resnet/tensorrt/calibrator.cache")
            preprocessed_data_dir = dict_get(self.args, "preprocessed_data_dir", default="build/preprocessed_data")
            calib_data_map = dict_get(self.args, "calib_data_map", default="data_maps/imagenet/cal_map.txt")
            calib_image_dir = os.path.join(preprocessed_data_dir, "imagenet/ResNet50/fp32")

            # Set up calibrator
            self.calibrator = RN50Calibrator(calib_batch_size=calib_batch_size, calib_max_batches=calib_max_batches,
                force_calibration=force_calibration, cache_file=cache_file,
                image_dir=calib_image_dir, calib_data_map=calib_data_map)
            self.builder_config.int8_calibrator = self.calibrator
            self.cache_file = cache_file
            self.need_calibration = force_calibration or not os.path.exists(cache_file)

    def initialize(self):
        # Create network.
        self.network = self.builder.create_network()

        # Parse from onnx file.
        parser = trt.OnnxParser(self.network, self.logger)
        with open(self.model_path, "rb") as f:
            model = f.read()
        success = parser.parse(model)
        if not success:
            raise RuntimeError("ResNet50 onnx model parsing failed! Error: {:}".format(parser.get_error(0).desc()))

        nb_layers = self.network.num_layers
        for i in range(nb_layers):
            layer = self.network.get_layer(i)
            # ':' in tensor names will screw up calibration cache parsing (which uses ':' as a delimiter)
            for j in range(layer.num_inputs):
                tensor = layer.get_input(j)
                tensor.name = tensor.name.replace(":", "_")
            for j in range(layer.num_outputs):
                tensor = layer.get_output(j)
                tensor.name = tensor.name.replace(":", "_")

        # Post-process the TRT network
        self.postprocess(useConvForFC = (self.precision == "int8"))
        if self.device_type == "gpu" and self.precision == "int8" and not self.need_calibration:
            # Read Calibration and fuse layers
            self.registry = trt.get_plugin_registry()
            parse_calibration(self.network, self.cache_file)
            fuse_br1_br2c_onnx(self.registry, self.network)
            fuse_br2b_br2c_onnx(self.registry, self.network)

        self.initialized = True

    def postprocess(self, useConvForFC=False):
        # Set input dtype and format
        input_tensor = self.network.get_input(0)
        if self.input_dtype == "int8":
            input_tensor.dtype = trt.int8
            input_tensor.dynamic_range = (-128, 127)
        if self.input_format == "linear":
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
        elif self.input_format == "chw4":
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.CHW4)

        # Get the layers we care about.
        nb_layers = self.network.num_layers
        logging.debug(nb_layers)
        for i in range(nb_layers):
            layer = self.network.get_layer(i)
            logging.debug("({:}) Layer '{:}' -> Type: {:} ON {:}".format(i, layer.name, layer.type,
                self.builder_config.get_device_type(layer)))

            # Detect the MM layer since we replace the MM layer with an FC layer.
            if "Matrix Multiply" in layer.name:
                # (i-1)th layer should be kernel.
                fc_kernel_layer = self.network.get_layer(i-1)
                assert "Constant" in fc_kernel_layer.name
                fc_kernel_layer.__class__ = trt.IConstantLayer
                fc_kernel = fc_kernel_layer.weights
                fc_kernel = fc_kernel.reshape(2048, 1001)
                fc_kernel = fc_kernel[:,1:]

                # (i-4)th layer should be reduction.
                reduce_layer = self.network.get_layer(i-4)
                assert "Reduce" in reduce_layer.name
                reduce_layer.__class__ = trt.IReduceLayer

                # (i-5)th layer should be the last ReLU
                last_conv_layer = self.network.get_layer(i-5)
                assert "Activation" in last_conv_layer.name
                last_conv_layer.__class__ = trt.IActivationLayer

                # (i+2)th layer should be the bias
                fc_bias_layer = self.network.get_layer(i+2)
                assert "Scale" in fc_bias_layer.name
                fc_bias_layer.__class__ = trt.IScaleLayer
                fc_bias = fc_bias_layer.shift[1:]

        # Unmark the old output since we are going to add new layers for the final part of the network.
        while self.network.num_outputs > 0:
            logging.info("Unmarking output: {:}".format(self.network.get_output(0).name))
            self.network.unmark_output(self.network.get_output(0))

        # Replace the reduce layer with pooling layer
        pool_layer_new = self.network.add_pooling(last_conv_layer.get_output(0), trt.PoolingType.AVERAGE, (7, 7))
        pool_layer_new.name = "squeeze_replaced"
        pool_layer_new.get_output(0).name = "squeeze_replaced_output"

        # Add fc layer
        fc_kernel = fc_kernel.reshape(2048, 1000).transpose().flatten()
        if useConvForFC:
            fc_layer_new = self.network.add_convolution(pool_layer_new.get_output(0), fc_bias.size, (1, 1), fc_kernel, fc_bias)
        else:
            fc_layer_new = self.network.add_fully_connected(pool_layer_new.get_output(0), fc_bias.size, fc_kernel, fc_bias)
        fc_layer_new.name = "fc_replaced"
        fc_layer_new.get_output(0).name = "fc_replaced_output"

        # Add topK layer.
        topk_layer = self.network.add_topk(fc_layer_new.get_output(0), trt.TopKOperation.MAX, 1, 1)
        topk_layer.name = "topk_layer"
        topk_layer.get_output(0).name = "topk_layer_output_value"
        topk_layer.get_output(1).name = "topk_layer_output_index"

        # Mark the new output.
        self.network.mark_output(topk_layer.get_output(1))

        if self.network.num_outputs != 1:
            logging.warning("num outputs should be 1 after unmarking! Has {:}".format(self.network.num_outputs))
            raise Exception

