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

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import os, sys
sys.path.insert(0, os.getcwd())

from code.common import logging

from PIL import Image

class SSDResNet34EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_dir, cache_file, batch_size, max_batches, force_calibration, calib_data_map):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file
        self.max_batches = max_batches

        image_list = []
        with open(calib_data_map) as f:
            for line in f:
                image_list.append(line.strip())

        self.shape = (batch_size, 3, 1200, 1200)
        self.device_input = cuda.mem_alloc(trt.volume(self.shape) * 4)

        self.coco_id = 0
        self.force_calibration = force_calibration

        # Create a generator that will give us batches. We can use next() to iterate over the result.
        def load_batches():
            batch_id = 0
            batch_size = self.shape[0]
            batch_data = np.zeros(shape=self.shape, dtype=np.float32)
            while self.coco_id < len(image_list) and batch_id < self.max_batches:
                print("Calibrating with batch {}".format(batch_id))
                batch_id += 1
                end_coco_id = min(self.coco_id + batch_size, len(image_list))

                for i in range(self.coco_id, end_coco_id):
                    batch_data[i - self.coco_id] = np.load(os.path.join(data_dir, image_list[i] + ".npy"))

                self.coco_id = end_coco_id

                shape = self.shape
                data = batch_data.tobytes()
                labels = bytes(b'')
                yield data

        self.batches = load_batches()

        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if not self.force_calibration and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                self.cache = f.read()
        else:
            self.cache = None

    def get_batch_size(self):
        return self.shape[0]

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        try:
            # Get a single batch.
            data = next(self.batches)
            # Copy to device, then return a list containing pointers to input device buffers.
            cuda.memcpy_htod(self.device_input, data)
            return [int(self.device_input)]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def read_calibration_cache(self):
        return self.cache

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def clear_cache(self):
        self.cache = None
