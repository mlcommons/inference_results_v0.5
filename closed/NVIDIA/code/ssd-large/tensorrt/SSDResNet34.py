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

import argparse
import ctypes
import os, sys

# The plugin .so file has to be loaded at global scope and before `import torch` to avoid cuda version mismatch.
NMS_OPT_PLUGIN_LIBRARY="build/plugins/NMSOptPlugin/libnmsoptplugin.so"
if not os.path.isfile(NMS_OPT_PLUGIN_LIBRARY):
    raise IOError("{}\n{}\n".format(
        "Failed to load library ({}).".format(NMS_OPT_PLUGIN_LIBRARY),
        "Please build the NMS Opt plugin."
    ))
ctypes.CDLL(NMS_OPT_PLUGIN_LIBRARY)

import numpy as np
import tensorrt as trt
sys.path.insert(0, os.getcwd())

from code.common import logging, dict_get
from code.common.builder import BenchmarkBuilder
from importlib import import_module
SSDResNet34EntropyCalibrator = import_module("code.ssd-large.tensorrt.calibrator").SSDResNet34EntropyCalibrator
load_torch_weights = import_module("code.ssd-large.tensorrt.utils").load_torch_weights
dboxes_R34_coco = import_module("code.ssd-large.tensorrt.utils").dboxes_R34_coco

import pycuda.driver as cuda
import pycuda.autoinit

INPUT_SHAPE = (3, 1200, 1200)

class SSDResNet34(BenchmarkBuilder):

    # Pop weights from the G_WEIGHTS dict, returning only the weights.
    # If the name does not match what is expected, returns None.
    # name mappings are:
    # Conv:
    #   kernel -> "weight"
    #   bias -> "bias"
    # BatchNorm:
    #   BN weight -> "weight"
    #   BN bias -> "bias"
    #   BN mean -> "running_mean"
    #   BN variance -> "running variance"
    def pop_weights(self, name):
        # The 1200x1200 .pth weights file has some extra parameters that are unused.
        # They are postfixed with 'num'batched_tracked'. Skip these weights if encoutered.
        extra_layer = "num_batches_tracked"
        next_weight_name = list(self.G_WEIGHTS.keys())[0]
        if (next_weight_name.find(extra_layer) > -1):
            self.G_WEIGHTS.popitem(last=False)[1].cpu().numpy()
            next_weight_name = list(self.G_WEIGHTS.keys())[0]
        if name not in next_weight_name:
            return None
        return self.G_WEIGHTS.popitem(last=False)[1].cpu().numpy()

    def get_nms_opt_plugin(self, plugin_name):
        plugin = None
        for plugin_creator in trt.get_plugin_registry().plugin_creator_list:
            if plugin_creator.name == plugin_name:
                # shareLocation = true.
                shareLocation_field = trt.PluginField("shareLocation", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32)
                # Encoded in target = false.
                varianceEncodedInTarget_field = trt.PluginField("varianceEncodedInTarget", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32)
                # backgroundLabelID = 0
                backgroundLabelId_field = trt.PluginField("backgroundLabelId", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32)
                # 81 classes
                numClasses_field = trt.PluginField("numClasses", np.array([81], dtype=np.int32), trt.PluginFieldType.INT32)
                # Top 200
                topK_field = trt.PluginField("topK", np.array([200], dtype=np.int32), trt.PluginFieldType.INT32)
                # Keep topK 200
                keepTopK_field = trt.PluginField("keepTopK", np.array([200], dtype=np.int32), trt.PluginFieldType.INT32)
                # confidence threshold = 0.05 from utils.py
                confidenceThreshold_field = trt.PluginField("confidenceThreshold", np.array([0.05], dtype=np.float32), trt.PluginFieldType.FLOAT32)
                # 0.5 as per criteria in infer.py
                nmsThreshold_field = trt.PluginField("nmsThreshold", np.array([0.5], dtype=np.float32), trt.PluginFieldType.FLOAT32)
                # input order, default to loc_data, confg_data, priorbox_data
                inputOrder_field = trt.PluginField("inputOrder", np.array([0,6,12], dtype=np.int32), trt.PluginFieldType.INT32)

                confSigmoid_field = trt.PluginField("confSigmoid", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32)
                confSoftmax_field = trt.PluginField("confSoftmax", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32)
                # are bbox data normalized by the network?
                isNormalized_field = trt.PluginField("isNormalized", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32)
                # coding method for bbox = 1: CENTER_SIZE
                codeType_field = trt.PluginField("codeType", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32)
                numLayers_field = trt.PluginField("numLayers", np.array([6], dtype=np.int32), trt.PluginFieldType.INT32)

                field_collection = trt.PluginFieldCollection(
                    [shareLocation_field,
                    varianceEncodedInTarget_field,
                    backgroundLabelId_field,
                    numClasses_field,
                    topK_field,
                    keepTopK_field,
                    confidenceThreshold_field,
                    nmsThreshold_field,
                    inputOrder_field,
                    confSigmoid_field,
                    confSoftmax_field,
                    isNormalized_field,
                    codeType_field,
                    numLayers_field])

                plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)
        return plugin

    # Adds a conv layer to the network, popping two weights off the top of the weights dictionary.
    def add_conv(self, network, inp, padding=None, stride=None, dilation=None):
        # Kernel should never be missing.
        kernel = self.pop_weights("weight")
        # Kernel is always NCHW
        kernel_N = kernel.shape[0]
        kernel_HW = kernel.shape[2:4]
        # Bias can be missing.
        bias = self.pop_weights("bias")
        bias = bias if bias is not None else trt.Weights()
        conv = network.add_convolution(inp, num_output_maps=kernel_N, kernel_shape=kernel_HW, kernel=kernel, bias=bias)
        conv.stride = stride or conv.stride
        conv.padding = padding or conv.padding
        conv.dilation = dilation or conv.dilation

        return conv.get_output(0)

    def add_batchnorm(self, network, inp):
        # All of these are expected to be present.
        weight = self.pop_weights("weight")
        bias = self.pop_weights("bias")
        running_mean = self.pop_weights("running_mean")
        running_var = self.pop_weights("running_var")
        # Add batchnorm implemented as a Scale layer.
        scale = weight / np.sqrt(running_var + 1e-5)
        shift = bias - scale * running_mean
        bn = network.add_scale(inp, trt.ScaleMode.CHANNEL, shift=shift, scale=scale)
        return bn.get_output(0)

    def add_conv_relu_bn_pair(self, network, inp, prefix, pads=None, strides=None, dilations=None):
        pads = pads or [(0,0), (0,0)]
        strides = strides or [(1,1), (1,1)]
        dilations = dilations or [(1,1), (1,1)]
        inp = self.add_conv(network, inp, pads[0], strides[0], dilations[0])
        inp = self.add_batchnorm(network, inp)
        relu = network.add_activation(inp, type=trt.ActivationType.RELU)
        inp = self.add_conv(network, relu.get_output(0), pads[1], strides[1], dilations[1])
        inp = self.add_batchnorm(network, inp)
        return inp

    def add_conv_relu_pair(self, network, inp, prefix, pads=None, strides=None, dilations=None):
        pads = pads or [(0,0), (0,0)]
        strides = strides or [(1,1), (1,1)]
        dilations = dilations or [(1,1), (1,1)]
        inp = self.add_conv(network, inp, pads[0], strides[0], dilations[0])
        relu = network.add_activation(inp, type=trt.ActivationType.RELU)
        inp = self.add_conv(network, relu.get_output(0), pads[1], strides[1], dilations[1])
        relu = network.add_activation(inp, type=trt.ActivationType.RELU)
        return relu.get_output(0)

    def populate_network(self, network):
        # TODO: Make the pth file path configurable.
        self.G_WEIGHTS = load_torch_weights(self.model_path)

        inp = network.add_input(name="input", shape=INPUT_SHAPE, dtype=trt.float32)
        pad3 = (3,3)
        stride2 = (2,2)
        stride3 = (3,3)
        # 1.0
        inp = self.add_conv(network, inp, pad3, stride2)
        # 1.1
        inp = self.add_batchnorm(network, inp)
        inp = network.add_activation(inp, type=trt.ActivationType.RELU)
        pool = network.add_pooling(input=inp.get_output(0), type=trt.PoolingType.MAX, window_size=(3, 3))
        pool.stride = (2,2)
        pool.padding = (1,1)
        inp = pool.get_output(0)

        # 1.4.0
        pad1 = [(1,1),(1,1)]
        inp = self.add_conv_relu_bn_pair(network, inp, "1.4.0", pad1)
        inp = network.add_elementwise(inp, pool.get_output(0), trt.ElementWiseOperation.SUM)
        relu = network.add_activation(inp.get_output(0), type=trt.ActivationType.RELU)

        # 1.4.1
        inp = self.add_conv_relu_bn_pair(network, relu.get_output(0), "1.4.1", pad1)
        inp = network.add_elementwise(inp, relu.get_output(0), trt.ElementWiseOperation.SUM)
        relu = network.add_activation(inp.get_output(0), type=trt.ActivationType.RELU)

        # 1.4.2
        inp = self.add_conv_relu_bn_pair(network, relu.get_output(0), "1.4.2", pad1)
        inp = network.add_elementwise(inp, relu.get_output(0), trt.ElementWiseOperation.SUM)
        relu = network.add_activation(inp.get_output(0), type=trt.ActivationType.RELU)

        # 1.5.0
        strides21 = [(2,2),(1,1)]
        inp = self.add_conv_relu_bn_pair(network, relu.get_output(0), "1.5.0", pad1, strides21)
        downsample_conv = self.add_conv(network, relu.get_output(0), None, (2,2))
        downsample_bn = self.add_batchnorm(network, downsample_conv)
        inp = network.add_elementwise(inp, downsample_bn, trt.ElementWiseOperation.SUM)
        relu = network.add_activation(inp.get_output(0), type=trt.ActivationType.RELU)

        # 1.5.1
        inp = self.add_conv_relu_bn_pair(network, relu.get_output(0), "1.5.1", pad1)
        inp = network.add_elementwise(inp, relu.get_output(0), trt.ElementWiseOperation.SUM)
        relu = network.add_activation(inp.get_output(0), type=trt.ActivationType.RELU)

        # 1.5.2
        inp = self.add_conv_relu_bn_pair(network, relu.get_output(0), "1.5.2", pad1)
        inp = network.add_elementwise(inp, relu.get_output(0), trt.ElementWiseOperation.SUM)
        relu = network.add_activation(inp.get_output(0), type=trt.ActivationType.RELU)

        # 1.5.3
        inp = self.add_conv_relu_bn_pair(network, relu.get_output(0), "1.5.3", pad1)
        inp = network.add_elementwise(inp, relu.get_output(0), trt.ElementWiseOperation.SUM)
        relu = network.add_activation(inp.get_output(0), type=trt.ActivationType.RELU)

        # 2.0.0
        inp = self.add_conv_relu_bn_pair(network, relu.get_output(0), "2.0.0", pad1)
        downsample_conv = self.add_conv(network, relu.get_output(0))
        downsample_bn = self.add_batchnorm(network, downsample_conv)
        inp = network.add_elementwise(inp, downsample_bn, trt.ElementWiseOperation.SUM)
        relu = network.add_activation(inp.get_output(0), type=trt.ActivationType.RELU)

        # 2.0.1
        inp = self.add_conv_relu_bn_pair(network, relu.get_output(0), "2.0.1", pad1)
        inp = network.add_elementwise(inp, relu.get_output(0), trt.ElementWiseOperation.SUM)
        relu = network.add_activation(inp.get_output(0), type=trt.ActivationType.RELU)

        # 2.0.2
        inp = self.add_conv_relu_bn_pair(network, relu.get_output(0), "2.0.2", pad1)
        inp = network.add_elementwise(inp, relu.get_output(0), trt.ElementWiseOperation.SUM)
        relu = network.add_activation(inp.get_output(0), type=trt.ActivationType.RELU)

        # 2.0.3
        inp = self.add_conv_relu_bn_pair(network, relu.get_output(0), "2.0.2", pad1)
        inp = network.add_elementwise(inp, relu.get_output(0), trt.ElementWiseOperation.SUM)
        relu = network.add_activation(inp.get_output(0), type=trt.ActivationType.RELU)

        # 2.0.4
        inp = self.add_conv_relu_bn_pair(network, relu.get_output(0), "2.0.4", pad1)
        inp = network.add_elementwise(inp, relu.get_output(0), trt.ElementWiseOperation.SUM)
        relu = network.add_activation(inp.get_output(0), type=trt.ActivationType.RELU)

        # 2.0.5
        inp = self.add_conv_relu_bn_pair(network, relu.get_output(0), "2.0.5", pad1)
        inp = network.add_elementwise(inp, relu.get_output(0), trt.ElementWiseOperation.SUM)
        relu = network.add_activation(inp.get_output(0), type=trt.ActivationType.RELU)

        # Start of last stage layers
        strides12 = [(1,1),(2,2)]
        pads01 = [(0,0),(1,1)]

        # additional_blocks.0.0
        relu0 = self.add_conv_relu_pair(network, relu.get_output(0), "additional_blocks.0.0", pads01, strides12)

        # additional_blocks.1.0
        relu1 = self.add_conv_relu_pair(network, relu0, "additional_blocks.1.0", pads01, strides12)

        # additional_blocks.2.0
        relu2 = self.add_conv_relu_pair(network, relu1, "additional_blocks.2.0", pads01, strides12)

        # additional_blocks.3.0
        relu3 = self.add_conv_relu_pair(network, relu2, "additional_blocks.3.0", strides=strides12)

        # additional_blocks.4.0
        relu4 = self.add_conv_relu_pair(network, relu3, "additional_blocks.4.0")

        # stride3 for 1200x1200 pytorch model.
        loc0 = self.add_conv(network, relu.get_output(0), (1,1), stride3)
        loc1 = self.add_conv(network, relu0, (1,1), stride3)
        loc2 = self.add_conv(network, relu1, (1,1), stride3)
        loc3 = self.add_conv(network, relu2, (1,1), stride3)
        loc4 = self.add_conv(network, relu3, (1,1), stride3)
        loc5 = self.add_conv(network, relu4, (1,1), stride3)

        conf0 = self.add_conv(network, relu.get_output(0), (1,1), stride3)
        conf1 = self.add_conv(network, relu0, (1,1), stride3)
        conf2 = self.add_conv(network, relu1, (1,1), stride3)
        conf3 = self.add_conv(network, relu2, (1,1), stride3)
        conf4 = self.add_conv(network, relu3, (1,1), stride3)
        conf5 = self.add_conv(network, relu4, (1,1), stride3)

        nms_opt_plugin = self.get_nms_opt_plugin("NMS_OPT_TRT")

        image_size = (1200, 1200)
        strides = [3,3,2,2,2,2]

        # Dboxes information - taken from dboxes_R34_coco from infer.py
        dboxes_val = dboxes_R34_coco(image_size, strides).dboxes_ltrb.numpy()
        dboxes_flatten_val = dboxes_val.reshape((15130*4,1))

        x = np.array([0.1, 0.1, 0.2, 0.2], dtype=np.float32)
        variances = np.tile(x, 15130).reshape(15130*4, 1)
        concat_dboxes_val = np.concatenate((dboxes_flatten_val, variances), axis=0)

        dboxes_reshaped_val = concat_dboxes_val.reshape((2, 15130*4, 1))
        dboxes_reshaped = network.add_constant((2, 15130*4, 1), dboxes_reshaped_val)

        nms_layer = network.add_plugin_v2([loc0, loc1, loc2,
                                    loc3, loc4, loc5,
                                    conf0, conf1, conf2,
                                    conf3, conf4, conf5,
                                    dboxes_reshaped.get_output(0)], nms_opt_plugin)
        # Top detections
        network.mark_output(nms_layer.get_output(0))
        nms_layer.get_output(0).name = "NMS_0"

        return network

    def __init__(self, args):
        super().__init__(args, name="ssd-large", workspace_size=(2<<30))

        # Model path
        self.model_path = dict_get(args, "model_path", default="build/models/SSDResNet34/resnet34-ssd1200.pytorch")

        if self.precision == "int8":
            force_calibration = dict_get(self.args, "force_calibration", default=False)
            calib_batch_size = dict_get(self.args, "calib_batch_size", default=10)
            calib_max_batches = dict_get(self.args, "calib_max_batches", default=50)
            cache_file = dict_get(self.args, "cache_file", default="code/ssd-large/tensorrt/calibrator.cache")
            preprocessed_data_dir = dict_get(self.args, "preprocessed_data_dir", default="build/preprocessed_data")
            calib_data_map = dict_get(self.args, "calib_data_map", default="data_maps/coco/cal_map.txt")
            calib_image_dir = os.path.join(preprocessed_data_dir, "coco/train2017/SSDResNet34/fp32")

            self.calibrator = SSDResNet34EntropyCalibrator(calib_image_dir, cache_file, calib_batch_size,
                calib_max_batches, force_calibration, calib_data_map)
            self.builder_config.int8_calibrator = self.calibrator
            self.cache_file = cache_file

    def initialize(self):
        # Create network.
        self.network = self.builder.create_network()

        # Populate network.
        self.populate_network(self.network)

        # Set input dtype and format
        input_tensor = self.network.get_input(0)
        if self.input_dtype == "int8":
            input_tensor.dtype = trt.int8
        if self.input_format == "linear":
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
        elif self.input_format == "chw4":
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.CHW4)

        # Fall back to GPU for the last conv layers (inputs to NMS plugin) for better accuracy.
        if self.dla_core is not None:
            for i in range(119,132):
                self.builder_config.set_device_type(self.network.get_layer(i), trt.DeviceType.GPU)

        self.initialized = True
