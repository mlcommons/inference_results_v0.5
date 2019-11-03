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

import ctypes
import os
import tensorrt as trt
import numpy as np
from code.resnet.tensorrt.network_search import network_search as ns
from code.common import logging

#
#  Usage:
#    parse_calibration(network, "calibrator.cache")
#
#    fuse_br1_br2c_uff(registry, network)
#    fuse_br2b_br2c_uff(registry, network)
#           --- or ---
#    fuse_br1_br2c_onnx(registry, network)
#    fuse_br2c_br2c_onnx(registry, network)
#

def parse_calibration(network, cache_path):
    # Parse the calibration file, set dynamic range on all network tensors
    if not os.path.exists(cache_path):
        return
    tensors = {}
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type != trt.LayerType.CONSTANT:
            for j in range(layer.num_inputs):
                tensor = layer.get_input(j)
                tensors[tensor.name] = tensor
            for j in range(layer.num_outputs):
                tensor = layer.get_output(j)
                tensors[tensor.name] = tensor
    np127=np.float32(127.0)
    with open(cache_path, "rb") as f:
        lines=f.read().decode('ascii').splitlines()
    for line in lines:
        split=line.split(':')
        if len(split)!=2:
            continue
        tensor = tensors[split[0]]
        if tensor is None:
            raise Exception("Tensor for name: " + split[0] + " not found")
        dynamic_range=np.uint32(int(split[1], 16)).view(np.dtype('float32')).item()*127.0
        tensor.set_dynamic_range(-dynamic_range, dynamic_range)
        # print("set " + split[0], dynamic_range)


def fuse_br1_br2c_uff(registry, network):
    pattern = [{"name":"input",  "type":trt.ITensor,               "children":["c_br1", "c_br2a"], "channels":64},
               {"name":"c_br1",  "type":trt.LayerType.CONVOLUTION, "children":"s_br1"},
               {"name":"s_br1",  "type":trt.LayerType.SCALE,       "children":"add"},
               {"name":"c_br2a", "type":trt.LayerType.CONVOLUTION, "children":"s_br2a"},
               {"name":"s_br2a", "type":trt.LayerType.SCALE,       "children":"r_br2a"},
               {"name":"r_br2a", "type":trt.LayerType.ACTIVATION,  "children":"c_br2b", "subtype":trt.ActivationType.RELU},
               {"name":"c_br2b", "type":trt.LayerType.CONVOLUTION, "children":"s_br2b"},
               {"name":"s_br2b", "type":trt.LayerType.SCALE,       "children":"r_br2b"},
               {"name":"r_br2b", "type":trt.LayerType.ACTIVATION,  "children":"c_br2c", "subtype":trt.ActivationType.RELU},
               {"name":"c_br2c", "type":trt.LayerType.CONVOLUTION, "children":"s_br2c"},
               {"name":"s_br2c", "type":trt.LayerType.SCALE,       "children":"add"},
               {"name":"add",    "type":trt.LayerType.ELEMENTWISE, "children":"relu",   "op":trt.ElementWiseOperation.SUM},
               {"name":"relu",   "type":trt.LayerType.ACTIVATION,  "children":"output", "subtype":trt.ActivationType.RELU},
               {"name":"output", "type":trt.ITensor,               "channels":256}]

    matches = ns.search(network, pattern)
    matchNumber = 0
    for match in matches:
        matchNumber = matchNumber + 1
        pluginName = "RES2_BR1_BR2C_" + str(matchNumber)

        # build an array with the dynamic ranges computed during calibration
        dynamic_ranges=np.array([match["input"].get_dynamic_range(),
                                 match["s_br1"].get_output(0).get_dynamic_range(),
                                 match["c_br2c"].get_input(0).get_dynamic_range(),
                                 match["s_br2c"].get_output(0).get_dynamic_range(),
                                 match["output"].get_dynamic_range()], dtype=np.float32)
                                 
        # build plugin fields, with weight/scale/bias/dynamic_range data
        fields = trt.PluginFieldCollection()
        fields.append(trt.PluginField("c_br1_w", match["c_br1"].kernel.data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("s_br1_s", match["s_br1"].scale.data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("s_br1_b", match["s_br1"].shift.data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("c_br2c_w", match["c_br2c"].kernel.data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("s_br2c_s", match["s_br2c"].scale.data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("s_br2c_b", match["s_br2c"].shift.data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("dynamic_ranges", memoryview(dynamic_ranges), trt.PluginFieldType.FLOAT32))
            
        creator=registry.get_plugin_creator('RnRes2Br1Br2c_TRT', '1', '');
        if creator is None:
            raise Exception("Creator for 'RnRes2Br1Br2c_TRT' not found")
        plugin=creator.create_plugin(pluginName, fields)
        if plugin is None:
            raise Exception("Plugin creation failed")
          
        logging.info("Plugin creation successful")
        inputs = [match["input"], match["r_br2b"].get_output(0)]
        layer = network.add_plugin_v2(inputs, plugin)
        layer.name = pluginName
        
        unfusedOutput = match["output"]
        fusedOutput = layer.get_output(0)
        fusedOutput.set_dynamic_range(-unfusedOutput.get_dynamic_range(), unfusedOutput.get_dynamic_range())
            
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.name==pluginName:
                continue
            for j in range(layer.num_inputs):
               if layer.get_input(j) == unfusedOutput:
                   layer.set_input(j, fusedOutput)

def fuse_br2b_br2c_uff(registry, network):
    pattern = [{"name":"input",  "type":trt.ITensor,               "children":["add", "c_br2a"], "channels":256},
               {"name":"c_br2a", "type":trt.LayerType.CONVOLUTION, "children":"s_br2a"},
               {"name":"s_br2a", "type":trt.LayerType.SCALE,       "children":"r_br2a"},
               {"name":"r_br2a", "type":trt.LayerType.ACTIVATION,  "children":"c_br2b", "subtype":trt.ActivationType.RELU},
               {"name":"c_br2b", "type":trt.LayerType.CONVOLUTION, "children":"s_br2b"},
               {"name":"s_br2b", "type":trt.LayerType.SCALE,       "children":"r_br2b"},
               {"name":"r_br2b", "type":trt.LayerType.ACTIVATION,  "children":"c_br2c", "subtype":trt.ActivationType.RELU},
               {"name":"c_br2c", "type":trt.LayerType.CONVOLUTION, "children":"s_br2c"},
               {"name":"s_br2c", "type":trt.LayerType.SCALE,       "children":"add"},
               {"name":"add",    "type":trt.LayerType.ELEMENTWISE, "children":"relu",   "op":trt.ElementWiseOperation.SUM},
               {"name":"relu",   "type":trt.LayerType.ACTIVATION,  "children":"output", "subtype":trt.ActivationType.RELU},
               {"name":"output", "type":trt.ITensor,               "channels":256}]

    matchNumber = 0
    while True:
        match = ns.search(network, pattern, True)
        if match==None:
            break
        """
        matches = ns.search(network, pattern)
        for match in matches:
        print (match["add"].name)
        """

        matchNumber = matchNumber + 1
        pluginName = "RES2_BR2B_BR2C_" + str(matchNumber)

        # build an array with the dynamic ranges computed during calibration
        dynamic_ranges=np.array([match["input"].get_dynamic_range(),
                                 match["c_br2b"].get_input(0).get_dynamic_range(),
                                 match["r_br2b"].get_output(0).get_dynamic_range(),
                                 match["s_br2c"].get_output(0).get_dynamic_range(),
                                 match["output"].get_dynamic_range()], dtype=np.float32)
                                 
        # build plugin fields, with weight/scale/bias/dynamic_range data
        fields = trt.PluginFieldCollection()
        fields.append(trt.PluginField("c_br2b_w", match["c_br2b"].kernel.data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("s_br2b_s", match["s_br2b"].scale.data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("s_br2b_b", match["s_br2b"].shift.data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("c_br2c_w", match["c_br2c"].kernel.data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("s_br2c_s", match["s_br2c"].scale.data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("s_br2c_b", match["s_br2c"].shift.data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("dynamic_ranges", memoryview(dynamic_ranges), trt.PluginFieldType.FLOAT32))
        
        creator=registry.get_plugin_creator('RnRes2Br2bBr2c_TRT', '1', '');
        if creator is None:
            raise Exception("Creator for 'RnRes2Br2bBr2c_TRT' not found")
        plugin=creator.create_plugin(pluginName, fields)
        if plugin is None:
            raise Exception("Plugin creation failed")
          
        logging.info("Plugin creation successful")
        inputs = [match["input"], match["r_br2a"].get_output(0)]
        #inputs=[match["input"]]
        layer = network.add_plugin_v2(inputs, plugin)
        layer.name = pluginName
        
        unfusedOutput = match["output"]
        fusedOutput = layer.get_output(0)
        fusedOutput.set_dynamic_range(-unfusedOutput.get_dynamic_range(), unfusedOutput.get_dynamic_range())
            
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.name==pluginName:
                continue
            for j in range(layer.num_inputs):
                if layer.get_input(j) == unfusedOutput:
                    logging.info("Updating input")
                    layer.set_input(j, fusedOutput)

def fuse_br1_br2c_onnx(registry, network):
    pattern = [{"name":"input",  "type":trt.ITensor,               "children":["c_br1", "c_br2a"], "channels":64},
               {"name":"c_br1",  "type":trt.LayerType.CONVOLUTION, "children":"add"},
               {"name":"c_br2a", "type":trt.LayerType.CONVOLUTION, "children":"r_br2a"},
               {"name":"r_br2a", "type":trt.LayerType.ACTIVATION,  "children":"c_br2b", "subtype":trt.ActivationType.RELU},
               {"name":"c_br2b", "type":trt.LayerType.CONVOLUTION, "children":"r_br2b"},
               {"name":"r_br2b", "type":trt.LayerType.ACTIVATION,  "children":"c_br2c", "subtype":trt.ActivationType.RELU},
               {"name":"c_br2c", "type":trt.LayerType.CONVOLUTION, "children":"add"},
               {"name":"add",    "type":trt.LayerType.ELEMENTWISE, "children":"relu",   "op":trt.ElementWiseOperation.SUM},
               {"name":"relu",   "type":trt.LayerType.ACTIVATION,  "children":"output", "subtype":trt.ActivationType.RELU},
               {"name":"output", "type":trt.ITensor,               "channels":256}]

    scale = trt.Weights(np.ones((256), dtype=np.float32))

    matches = ns.search(network, pattern)
    matchNumber = 0
    for match in matches:
        matchNumber = matchNumber + 1
        pluginName = "RES2_BR1_BR2C_" + str(matchNumber)

        # build an array with the dynamic ranges computed during calibration
        dynamic_ranges=np.array([match["input"].get_dynamic_range(),
                                 match["c_br1"].get_output(0).get_dynamic_range(),
                                 match["c_br2c"].get_input(0).get_dynamic_range(),
                                 match["c_br2c"].get_output(0).get_dynamic_range(),
                                 match["output"].get_dynamic_range()], dtype=np.float32)


        logging.info(dynamic_ranges)                             
        # build plugin fields, with weight/scale/bias/dynamic_range data
        fields = trt.PluginFieldCollection()
        fields.append(trt.PluginField("c_br1_w", match["c_br1"].kernel.data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("s_br1_s", scale.numpy().data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("s_br1_b", match["c_br1"].bias.data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("c_br2c_w", match["c_br2c"].kernel.data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("s_br2c_s", scale.numpy().data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("s_br2c_b", match["c_br2c"].bias.data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("dynamic_ranges", memoryview(dynamic_ranges), trt.PluginFieldType.FLOAT32))
            
        creator=registry.get_plugin_creator('RnRes2Br1Br2c_TRT', '1', '');
        if creator is None:
            raise Exception("Creator for 'RnRes2Br1Br2c_TRT' not found")
        plugin=creator.create_plugin(pluginName, fields)
        if plugin is None:
            raise Exception("Plugin creation failed")
          
        logging.info("Plugin creation successful")
        inputs = [match["input"], match["r_br2b"].get_output(0)]
        #inputs = [match["output"]]
        layer = network.add_plugin_v2(inputs, plugin)
        layer.name = pluginName
        
        unfusedOutput = match["output"]
        fusedOutput = layer.get_output(0)
        fusedOutput.set_dynamic_range(-unfusedOutput.get_dynamic_range(), unfusedOutput.get_dynamic_range())
            
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.name==pluginName:
                continue
            for j in range(layer.num_inputs):
                if layer.get_input(j) == unfusedOutput:
                    layer.set_input(j, fusedOutput)

def fuse_br2b_br2c_onnx(registry, network):
    pattern = [{"name":"input",  "type":trt.ITensor,               "children":["add", "c_br2a"], "channels":256},
               {"name":"c_br2a", "type":trt.LayerType.CONVOLUTION, "children":"r_br2a"},
               {"name":"r_br2a", "type":trt.LayerType.ACTIVATION,  "children":"c_br2b", "subtype":trt.ActivationType.RELU},
               {"name":"c_br2b", "type":trt.LayerType.CONVOLUTION, "children":"r_br2b"},
               {"name":"r_br2b", "type":trt.LayerType.ACTIVATION,  "children":"c_br2c", "subtype":trt.ActivationType.RELU},
               {"name":"c_br2c", "type":trt.LayerType.CONVOLUTION, "children":"add"},
               {"name":"add",    "type":trt.LayerType.ELEMENTWISE, "children":"relu",   "op":trt.ElementWiseOperation.SUM},
               {"name":"relu",   "type":trt.LayerType.ACTIVATION,  "children":"output", "subtype":trt.ActivationType.RELU},
               {"name":"output", "type":trt.ITensor,               "channels":256}]

    scale64 = trt.Weights(np.ones((64), dtype=np.float32))
    scale256 = trt.Weights(np.ones((256), dtype=np.float32))

    matchNumber = 0
    while True:
        match = ns.search(network, pattern, True)
        if match==None:
            break
        """
        matches = ns.search(network, pattern)
        for match in matches:
        print (match["add"].name)
        """

        matchNumber = matchNumber + 1
        pluginName = "RES2_BR2B_BR2C_" + str(matchNumber)

        # build an array with the dynamic ranges computed during calibration
        dynamic_ranges=np.array([match["input"].get_dynamic_range(),
                                 match["c_br2b"].get_input(0).get_dynamic_range(),
                                 match["r_br2b"].get_output(0).get_dynamic_range(),
                                 match["c_br2c"].get_output(0).get_dynamic_range(),
                                 match["output"].get_dynamic_range()], dtype=np.float32)
                                 
        # build plugin fields, with weight/scale/bias/dynamic_range data
        fields = trt.PluginFieldCollection()
        fields.append(trt.PluginField("c_br2b_w", match["c_br2b"].kernel.data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("s_br2b_s", scale64.numpy().data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("s_br2b_b", match["c_br2b"].bias.data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("c_br2c_w", match["c_br2c"].kernel.data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("s_br2c_s", scale256.numpy().data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("s_br2c_b", match["c_br2c"].bias.data, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("dynamic_ranges", memoryview(dynamic_ranges), trt.PluginFieldType.FLOAT32))
        
        creator=registry.get_plugin_creator('RnRes2Br2bBr2c_TRT', '1', '');
        if creator is None:
            raise Exception("Creator for 'RnRes2Br2bBr2c_TRT' not found")
        plugin=creator.create_plugin(pluginName, fields)
        if plugin is None:
            raise Exception("Plugin creation failed")
          
        logging.info("Plugin creation successful")
        inputs = [match["input"], match["r_br2a"].get_output(0)]
        #inputs=[match["input"]]
        layer = network.add_plugin_v2(inputs, plugin)
        layer.name = pluginName
        
        unfusedOutput = match["output"]
        fusedOutput = layer.get_output(0)
        fusedOutput.set_dynamic_range(-unfusedOutput.get_dynamic_range(), unfusedOutput.get_dynamic_range())
            
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.name==pluginName:
                continue
            for j in range(layer.num_inputs):
                if layer.get_input(j) == unfusedOutput:
                    logging.info("Updating input")
                    layer.set_input(j, fusedOutput)
