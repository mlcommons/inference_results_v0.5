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
sys.path.append(os.getcwd())

import ctypes
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import pytest
import tensorrt as trt
import time

from code.common import logging
from glob import glob

class HostDeviceMem(object):
    def __init__(self, host, device):
        self.host = host
        self.device = device

def allocate_buffers(engine):
    d_inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        logging.info("Binding {:}".format(binding))
        dtype = engine.get_binding_dtype(binding)
        format = engine.get_binding_format(engine.get_binding_index(binding))
        shape = engine.get_binding_shape(binding)
        if format == trt.TensorFormat.CHW4:
            shape[-3] = ((shape[-3] - 1) // 4 + 1) * 4
        size = trt.volume(shape) * engine.max_batch_size
        # Allocate host and device buffers
        device_mem = cuda.mem_alloc(size * dtype.itemsize)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            d_inputs.append(device_mem)
        else:
            host_mem = cuda.pagelocked_empty(size, trt.nptype(dtype))
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return d_inputs, outputs, bindings, stream

def get_input_format(engine):
    return engine.get_binding_dtype(0), engine.get_binding_format(0)

class EngineRunner():

    def __init__(self, engine_file, verbose=False, plugins=None):
        self.engine_file = engine_file
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
        if not os.path.exists(engine_file):
            raise ValueError("File {:} does not exist".format(engine_file))

        trt.init_libnvinfer_plugins(self.logger, "")
        if plugins is not None:
            for plugin in plugins:
                ctypes.CDLL(plugin)
        self.engine = self.load_engine(engine_file)

        self.d_inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context()

    def load_engine(self, src_path):
        with open(src_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            buf = f.read()
            engine = runtime.deserialize_cuda_engine(buf)
        return engine

    def __call__(self, inputs, batch_size=1):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(d_input, inp, self.stream) for (d_input, inp) in zip(self.d_inputs, inputs)]
        # Run inference.
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in self.outputs]

    def __del__(self):
        # Clean up everything.
        with self.engine, self.context:
            [d_input.free() for d_input in self.d_inputs]
            [out.device.free() for out in self.outputs]
            del self.stream
