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
from code.common import logging

class network_search(object):
    def __init__(self, network):
        self.tensors = set()
        self.tensorReads = {}     # map:  tensor -> list of layers (read)
        self.layerWrites = {}     # map:  layer -> list of tensors (written)
        
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type != trt.LayerType.CONSTANT:
                writes = []
                self.layerWrites[layer] = writes
                for i in range(layer.num_outputs):
                    tensor = layer.get_output(i)
                    self.tensors.add(tensor)
                    writes.append(tensor)
                for i in range(layer.num_inputs):
                    tensor = layer.get_input(i)
                    self.tensors.add(tensor)
                    reads = self.tensorReads.get(tensor)
                    if reads is None:
                        reads = [layer]
                        self.tensorReads[tensor] = reads
                    else:
                        reads.append(layer)
                for tensor in self.tensors:
                    if self.tensorReads.get(tensor) is None:
                        self.tensorReads[tensor] = []

    @staticmethod
    def print_match(pattern, match):
        for node in pattern:
          key = node["name"]
          value = match[key]
          if isinstance(value, trt.ILayer):
              logging.info(key + "=" + match[key].name)
          else:
              logging.info(key + "=" + value.__str__())
    
    @staticmethod
    def print_matches(pattern, matches):
        matchNumber = 1
        if isinstance(matches, list):
            for match in matches:
                logging.info("Match number:", matchNumber)
                network_search.print_match(pattern, match)
                logging.info()
                matchNumber = matchNumber+1
        else:
            print_match(pattern + "=" + match)
    
    def match_tensor(self, tensor, values):
        channels = values.get("channels")
        if channels is not None:
            if len(tensor.shape)==0:
                return False
            if channels!=tensor.shape[0]:
                return False
        return True
    
    def match_convolution_layer(self, convolution_layer, values):
        return True

    def match_scale_layer(self, scale_layer, values):
        return True
      
    def match_activation_layer(self, activation_layer, values):
        subtype = values.get("subtype")
        if subtype is not None and subtype!=activation_layer.type:
            return False
        return True
    
    def match_element_wise_layer(self, element_wise_layer, values):
        op = values.get("op")
        if op is not None and op!=element_wise_layer.op:
            return False
        return True

    def match(self, current, search, key, state):
        entry = search[key]
        type = entry["type"]
        
        if isinstance(current, trt.ITensor):
            if isinstance(type, trt.LayerType):
                if len(self.tensorReads[current])!=1:
                    return False
                return self.match(self.tensorReads[current][0], search, key, state)
            else:
                if not self.match_tensor(current, entry):
                    return False
                children = entry.get("children")
                if children is not None:
                    if isinstance(children, str):
                        children = [children]
                    if not self.pair_match(self.tensorReads[current], search, children, state):
                        return False
                # fall through
        elif isinstance(current, trt.ILayer):
            current.__class__ = trt.ILayer
            layerType = current.type
            if not isinstance(type, trt.LayerType) or layerType!=type:
              return False
            
            #
            # For this example, I only need to match a few layer types, if more are required, please extend
            #
            if layerType==trt.LayerType.CONVOLUTION:
                current.__class__ = trt.IConvolutionLayer;
                if not self.match_convolution_layer(current, entry):
                    return False
            elif layerType==trt.LayerType.SCALE:
                current.__class__ = trt.IScaleLayer
                if not self.match_scale_layer(current, entry):
                    return False
            elif layerType==trt.LayerType.ACTIVATION:
                current.__class__ = trt.IActivationLayer
                if not self.match_activation_layer(current, entry):
                    return False
            elif layerType==trt.LayerType.ELEMENTWISE:
                current.__class__ = trt.IElementWiseLayer
                if not self.match_element_wise_layer(current, entry):
                    return False
            else:
                raise Exception("Layer type not implemented")
              
            children = entry.get("children")
            if children is not None:
                if isinstance(children, str):
                    children = [children];
                if not self.pair_match(self.layerWrites[current], search, children, state):
                    return False
            # fall through
        else:
            raise Exception("Unexpected type: " + current.__class__.__name__)

        join = state.get(key)
        if join is None:
            state[key] = current
        else:
            if join!=current:
                return False      
        return True

    def pair_match(self, currentList, search, keyList, state):
        # each "key" criteria must uniquely match exactly one "current", i.e., a bijection from keys to currents.
        
        if len(currentList)!=len(keyList):
            return False
        matchSet = set()
        bijectionMap = {}
        for key in keyList:
            count = 0
            for current in currentList:
                copy = state.copy()
                if self.match(current, search, key, copy):
                    count = count+1
                    matchSet.add(current)
                    bijectionMap[key] = current
            if count!=1:
                return False
        if len(matchSet)!=len(currentList):
            return False
        for key,current in bijectionMap.items():
            self.match(current, search, key, state)
        return True
      
    @staticmethod
    def search(network, pattern, singleMatch = False):
        engine = network_search(network)
        patternDictionary = {}
        for entry in pattern:
            patternDictionary[entry["name"]] = entry
        logging.info(patternDictionary)
        results = []
        for tensor in engine.tensors:
            state = {}
            if engine.match(tensor, patternDictionary, "input", state):
                if singleMatch:
                    return state
                results.append(state)
        if singleMatch:
            return None
        return results
      
