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

import struct
import tensorrt as trt
import os, sys
sys.path.insert(0, os.getcwd())

from code.common import logging, dict_get
from code.common.builder import BenchmarkBuilder
from importlib import import_module
MobileNetCalibrator = import_module("code.mobilenet.tensorrt.calibrator").MobileNetCalibrator

class MobileNet(BenchmarkBuilder):

    def __init__(self, args):
        super().__init__(args, name="mobilenet")

        # Model path
        self.model_path = dict_get(args, "model_path", default="build/models/MobileNet/mobilenet_sym_no_bn.onnx")

        # Get calibrator variables
        calib_batch_size = dict_get(self.args, "calib_batch_size", default=1)
        calib_max_batches = dict_get(self.args, "calib_max_batches", default=500)
        force_calibration = dict_get(self.args, "force_calibration", default=False)
        cache_file = dict_get(self.args, "cache_file", default="code/mobilenet/tensorrt/calibrator.cache")
        preprocessed_data_dir = dict_get(self.args, "preprocessed_data_dir", default="build/preprocessed_data")
        calib_data_map = dict_get(self.args, "calib_data_map", default="data_maps/imagenet/cal_map.txt")
        calib_image_dir = os.path.join(preprocessed_data_dir, "imagenet/MobileNet/fp32")
        # Set up calibrator
        self.calibrator = MobileNetCalibrator(calib_batch_size=calib_batch_size, calib_max_batches=calib_max_batches,
            force_calibration=force_calibration, cache_file=cache_file,
            image_dir=calib_image_dir, calib_data_map=calib_data_map)
        self.need_calibration = force_calibration or not os.path.exists(cache_file)
        self.cache_file = cache_file

    def initialize(self):
        # Only use calibration in calibration mode.
        if self.need_calibration:
            self.builder_config.int8_calibrator = self.calibrator
            self.explicit_precision = False

        # Create network.
        if self.precision == "int8" and self.explicit_precision and self.dla_core is None:
            network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
            self.network = self.builder.create_network(network_creation_flag)
            self.apply_flag(trt.BuilderFlag.STRICT_TYPES)
        else:
            self.network = self.builder.create_network()

        # Parse ONNX model and populate network.
        parser = trt.OnnxParser(self.network, self.logger)
        with open(self.model_path, "rb") as f:
            model = f.read()
        success = parser.parse(model)
        if not success:
            raise RuntimeError("MobileNet onnx model parsing failed! Error: {:}".format(parser.get_error(0).desc()))

        # Post-process the TRT network
        if self.precision == "fp16":
            # Due to the dynamic range limit, we need to scale down activations to avoid overflow in FP16 precision.
            self.postprocess(scale_factor=1000.0)
        elif self.precision == "int8":
            self.postprocess(useConvForFC=True)
        else:
            self.postprocess()

        if self.precision == "int8" and not self.need_calibration:
            # Use "explicit precision" feature in TensorRT 6 to load prequantized model.
            if self.explicit_precision and self.dla_core is None:
                self.set_layer_precision()
            else:
            # DLA does not support explicit precision, so fall back to scale settings.
                self.apply_custom_scales()

        self.initialized = True

    def postprocess(self, scale_factor=1.0, useConvForFC=False):
        # Set input dtype and format
        input_tensor = self.network.get_input(0)
        if self.input_dtype == "int8":
            input_tensor.dtype = trt.int8
            if self.dla_core is not None or not self.explicit_precision:
                input_tensor.dynamic_range = (-127, 127)
        if self.input_format == "linear":
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
        elif self.input_format == "chw4":
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.CHW4)

        # Get the layers we care about.
        nb_layers = self.network.num_layers
        first_scale = False
        logging.debug(nb_layers)
        for i in range(nb_layers):
            layer = self.network.get_layer(i)
            logging.debug("({:}) Layer '{:}' -> Type: {:}".format(i, layer.name, layer.type))
            # Detect the MM layer since we replace the MM layer with an FC layer.
            if "Matrix Multiply" in layer.name:
                # (i-1)th layer should be kernel.
                fc_kernel_layer = self.network.get_layer(i-1)
                assert "Constant" in fc_kernel_layer.name
                fc_kernel_layer.__class__ = trt.IConstantLayer
                fc_kernel = fc_kernel_layer.weights

                # (i-2)th layer should be bias.
                fc_bias_layer = self.network.get_layer(i-2)
                assert "Constant" in fc_bias_layer.name
                fc_bias_layer.__class__ = trt.IConstantLayer
                fc_bias = fc_bias_layer.weights

                # (i-4)th layer should be pooling.
                pool_layer = self.network.get_layer(i-4)
                assert "Pooling" in pool_layer.name
                pool_layer.__class__ = trt.IPoolingLayer

                # (i-5)th layer should be padding.
                padding_layer = self.network.get_layer(i-5)
                assert "Padding" in padding_layer.name
                padding_layer.__class__ = trt.IPaddingLayer
                # This padding layer should be a no-op
                assert (padding_layer.pre_padding == (0,0)) and (padding_layer.post_padding == (0,0))

                # (i-6)th layer. We don't care about the layer type, but we just need to reset output
                pre_padding_layer = self.network.get_layer(i-6)

            # Scale the weights of the first conv and the bias of all conv if scale is provided.
            if scale_factor != 1.0:
                if "Convolution" in layer.name:
                    conv_layer = layer
                    conv_layer.__class__ = trt.IConvolutionLayer
                    conv_layer.bias /= scale_factor
                    if not first_scale:
                        conv_layer.kernel /= scale_factor
                        first_scale = True

        # Remove the no-op padding layer by skipping to a copy of the pooling layer
        pool_layer_new = self.network.add_pooling(pre_padding_layer.get_output(0), pool_layer.type, pool_layer.window_size)
        pool_layer_new.name = "pooling_replaced"
        pool_layer_new.get_output(0).name = "pooling_replaced_output"

        # Unmark the old output since we are going to add new layers for the final part of the network.
        self.network.unmark_output(self.network.get_output(0))

        # Add fc layer.
        fc_kernel = fc_kernel.reshape(pool_layer_new.get_output(0).shape[0], fc_bias.size).transpose().flatten()
        fc_bias /= scale_factor
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

    def set_layer_precision(self):
        network = self.network
        # Set initial precision requirements for all layers and outputs tensors.
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)

        # Overwrite precision requirements specific to MobilenetV1
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            # Execution precision for all convolution layers should be INT8
            if layer.type == trt.LayerType.CONVOLUTION:
                layer.precision = trt.int8

            # Set Execution Precision for Quantizing/Dequantizing nodes
            if layer.type == trt.LayerType.SCALE:
                layer.precision = trt.float32
                # set output type of scale layer to INT8 as it is a quantizing scale
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.int8)

                # set input type of scale layer to FP32 as its a QUANTIZING NODE
                prev_layer = network.get_layer(i-1)
                # get all the inputs to scale layer, match with prev layer output
                for j in range(layer.num_inputs):
                    input_to_scale_layer = layer.get_input(j)
                    # for each input, check prev layer output
                    for k in range(prev_layer.num_outputs):
                        if prev_layer.get_output(k) == input_to_scale_layer:
                            prev_layer.set_output_type(k, trt.float32)

            # Set precision for ACTIVATION nodes to allow graph fusions, and for pooling nodes to run faster
            if layer.type == trt.LayerType.ACTIVATION or layer.type == trt.LayerType.POOLING:
                layer.precision = trt.int8
                layer.set_output_type(0, trt.int8)

            # Set the index output of topK layer to int32.
            if layer.type == trt.LayerType.TOPK:
                layer.set_output_type(1, trt.int32)

    def apply_custom_scales(self):
        def hex2float(s):
            return struct.unpack("!f", bytes.fromhex(s))[0]

        with open(self.cache_file) as f:
            contents = f.read()

        scale_cache = dict()
        for row in contents.split("\n")[1:]: # Skip the first row
            v = row.split(": ")
            if len(v) == 2:
                scale_cache[v[0]] = v[1]

        # Since the model is prequantized, all activations will be in range (-127.0, 127.0), except for the
        # last FC layer where we read scales from calibration cache.
        # Custom scale definitions:
        scales = {
            "pooling_replaced": ([scale_cache["164"]], [scale_cache["pooling_replaced_output"]]),
            "fc_replaced": ([scale_cache["pooling_replaced_output"]], [scale_cache["fc_replaced_output"]]),
            "topk_layer": ([scale_cache["fc_replaced_output"]],
                [scale_cache["topk_layer_output_value"], scale_cache["topk_layer_output_index"]])
        }

        for n in range(self.network.num_layers):
            layer = self.network.get_layer(n)

            if layer.name in scales:
                in_dyn_range = [(-127 * hex2float(i), 127 * hex2float(i)) for i in scales[layer.name][0]]
                out_dyn_range = [(-127 * hex2float(i), 127 * hex2float(i)) for i in scales[layer.name][1]]
            else:
                in_dyn_range = [(-127.0, 127.0) for i in range(layer.num_inputs)]
                out_dyn_range = [(-127.0, 127.0) for i in range(layer.num_outputs)]

            for i in range(layer.num_inputs):
                tensor = layer.get_input(i)
                tensor.dynamic_range = in_dyn_range[i]
            for i in range(layer.num_outputs):
                tensor = layer.get_output(i)
                tensor.dynamic_range = out_dyn_range[i]
