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
import json
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

import graphsurgeon as gs
import tensorflow as tf
import tensorrt as trt
import uff
sys.path.insert(0, os.getcwd())

from code.common import logging, dict_get
from code.common.builder import BenchmarkBuilder
from importlib import import_module
SSDMobileNetEntropyCalibrator = import_module("code.ssd-small.tensorrt.calibrator").SSDMobileNetEntropyCalibrator

Input = gs.create_node("Input",
    op="Placeholder",
    dtype=tf.float32,
    shape=[1, 3, 300, 300])
PriorBox = gs.create_plugin_node(name="MultipleGridAnchorGenerator", op="GridAnchor_TRT",
    numLayers=6,
    minSize=0.2,
    maxSize=0.95,
    aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
    variance=[0.1,0.1,0.2,0.2],
    featureMapShapes=[19, 10, 5, 3, 2, 1])
Postprocessor = gs.create_plugin_node(name="Postprocessor", op="NMS_OPT_TRT",
    shareLocation=1,
    varianceEncodedInTarget=0,
    backgroundLabelId=0,
    confidenceThreshold=0.3,
    nmsThreshold=0.6,
    topK=100,
    keepTopK=100,
    numClasses=91,
    inputOrder=[0, 7, 6],
    confSigmoid=1,
    confSoftmax=0,
    isNormalized=1,
    numLayers=6)
concat_priorbox = gs.create_node(name="concat_priorbox", op="ConcatV2", dtype=tf.float32, axis=2)
#concat_box_loc = gs.create_plugin_node("concat_box_loc", op="FlattenConcat_TRT", dtype=tf.float32, axis=1, ignoreBatch=0)
#concat_box_conf = gs.create_plugin_node("concat_box_conf", op="FlattenConcat_TRT", dtype=tf.float32, axis=1, ignoreBatch=0)

namespace_plugin_map = {
    "MultipleGridAnchorGenerator/Concatenate": concat_priorbox,
    "MultipleGridAnchorGenerator": PriorBox,
    "Postprocessor": Postprocessor,
    "image_tensor": Input,
    "ToFloat": Input,
    "Preprocessor": Input,
#    "concat": concat_box_loc,
#    "concat_1": concat_box_conf
}

def preprocess(dynamic_graph):
    dynamic_graph.forward_inputs(dynamic_graph.find_nodes_by_op("Identity"))
    dynamic_graph.forward_inputs(dynamic_graph.find_nodes_by_path("Squeeze"))
    dynamic_graph.forward_inputs(dynamic_graph.find_nodes_by_path("concat"))
    dynamic_graph.forward_inputs(dynamic_graph.find_nodes_by_path("concat_1"))

    for i in range(0,6):
        dynamic_graph.remove(dynamic_graph.find_nodes_by_path("BoxPredictor_{}/stack".format(i)))
        dynamic_graph.forward_inputs(dynamic_graph.find_nodes_by_path("BoxPredictor_{}/Reshape".format(i)))
        dynamic_graph.remove(dynamic_graph.find_nodes_by_path("BoxPredictor_{}/stack_1".format(i)))
        dynamic_graph.forward_inputs(dynamic_graph.find_nodes_by_path("BoxPredictor_{}/Reshape_1".format(i)))
        dynamic_graph.remove(dynamic_graph.find_nodes_by_path("BoxPredictor_{}/Shape".format(i)))

    # Now create a new graph by collapsing namespaces
    dynamic_graph.collapse_namespaces(namespace_plugin_map)
    # Remove the outputs, so we just have a single output node (Postprocessor).
    dynamic_graph.remove(dynamic_graph.graph_outputs, remove_exclusive_dependencies=False)
    # Disconnect the Input node from NMS.
    dynamic_graph.find_nodes_by_op("NMS_OPT_TRT")[0].input.remove("Input")
    # Disconnect concat/axis and concat_1/axis from NMS.
    dynamic_graph.find_nodes_by_op("NMS_OPT_TRT")[0].input.remove("concat/axis")
    dynamic_graph.find_nodes_by_op("NMS_OPT_TRT")[0].input.remove("concat_1/axis")
    dynamic_graph.find_nodes_by_name("Input")[0].input.remove("image_tensor:0")

class SSDMobileNet(BenchmarkBuilder):

    def __init__(self, args):
        super().__init__(args, name="ssd-small", workspace_size=(2<<30))

        # Model path
        self.model_path = dict_get(args, "model_path", default="build/models/SSDMobileNet/frozen_inference_graph.pb")

        if self.precision == "int8":
            calib_batch_size = dict_get(self.args, "calib_batch_size", default=1)
            calib_max_batches = dict_get(self.args, "calib_max_batches", default=500)
            force_calibration = dict_get(self.args, "force_calibration", default=False)
            cache_file = dict_get(self.args, "cache_file", default="code/ssd-small/tensorrt/calibrator.cache")
            preprocessed_data_dir = dict_get(self.args, "preprocessed_data_dir", default="build/preprocessed_data")
            calib_data_map = dict_get(self.args, "calib_data_map", default="data_maps/coco/cal_map.txt")
            calib_image_dir = os.path.join(preprocessed_data_dir, "coco/train2017/SSDMobileNet/fp32")

            self.calibrator = SSDMobileNetEntropyCalibrator(calib_batch_size, calib_max_batches,
                force_calibration, cache_file, calib_image_dir, calib_data_map)
            self.builder_config.int8_calibrator = self.calibrator
            self.cache_file = cache_file

    def initialize(self):
        # Create network.
        self.network = self.builder.create_network()

        # Do graph surgery on pb graph and convert to UFF.
        uff_model = uff.from_tensorflow_frozen_model(self.model_path, preprocessor="code/ssd-small/tensorrt/SSDMobileNet.py")

        # Parse UFF model and populate network.
        parser = trt.UffParser()
        parser.register_input("Input", [3, 300, 300], trt.UffInputOrder.NCHW)
        parser.register_output("Postprocessor")
        success = parser.parse_buffer(uff_model, self.network)
        if not success:
            raise RuntimeError("SSDMobileNet network creation failed!")

        # Set input dtype and format
        input_tensor = self.network.get_input(0)
        if self.input_dtype == "int8":
            input_tensor.dtype = trt.int8
            input_tensor.dynamic_range = (-1.0, 1.0)
        if self.input_format == "linear":
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
        elif self.input_format == "chw4":
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.CHW4)

        self.postprocess(replace_relu6=(self.dla_core is not None))

        self.initialized = True

    def postprocess(self, replace_relu6=False):
        nb_layers = self.network.num_layers

        # Layer preprocessing
        for i in range(nb_layers):
            layer = self.network.get_layer(i)
            logging.debug("({:}) Layer '{:}' -> Type: {:} ON {:}".format(i, layer.name, layer.type,
                self.builder_config.get_device_type(layer)))

            if replace_relu6 and "Relu6" in layer.name:
                activation = layer
                activation.__class__ = trt.IActivationLayer
                logging.debug("\tType: {:}, alpha={:}, beta={:}".format(activation.type, activation.alpha, activation.beta))
                # Convert to RELU
                if activation.type == trt.ActivationType.CLIP:
                    logging.debug("\tConverting to ReLU activation")
                    activation.type = trt.ActivationType.RELU
