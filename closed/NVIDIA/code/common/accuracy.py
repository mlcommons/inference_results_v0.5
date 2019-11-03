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

import time
import numpy as np
from code.common import logging

class AccuracyRunner(object):
    def __init__(self, runner, val_map, image_dir, verbose=False):
        self.runner = runner
        self.val_map = val_map
        self.image_dir = image_dir
        self.verbose = verbose

        self.image_list = []
        self.class_list = []

    def reset(self):
        self.image_list = []
        self.class_list = []

    def load_val_images(self):
        self.reset()
        with open(self.val_map) as f:
            for line in f:
                self.image_list.append(line.split()[0])
                self.class_list.append(int(line.split()[1]))

    def run(self):
        raise NotImplementedError("AccuracyRunner.run() is not implemented")

class ImageNetAccuracyRunner(AccuracyRunner):
    def __init__(self, runner, batch_size, image_dir, num_images, verbose=False):
        super().__init__(runner, "data_maps/imagenet/val_map.txt", image_dir, verbose=verbose)

        self.batch_size = batch_size
        self.num_images = num_images

    def run(self):
        self.load_val_images()
        logging.info("Running accuracy check on {:} images.".format(self.num_images))

        class_predictions = []
        batch_idx = 0
        for image_idx in range(0, self.num_images, self.batch_size):
            actual_batch_size = self.batch_size if image_idx + self.batch_size <= self.num_images else self.num_images - image_idx
            batch_images = self.image_list[image_idx:image_idx + actual_batch_size]
            # DLA does not support batches that are less than the engine's configured batch size. Pad with junk.
            while len(batch_images) < self.batch_size:
                batch_images.append(self.image_list[0])
            batch_images = np.ascontiguousarray(np.stack([np.load(os.path.join(self.image_dir, name + ".npy")) for name in batch_images]))

            start_time = time.time()
            outputs = self.runner([batch_images], self.batch_size)
            if self.verbose:
                logging.info("Batch {:d} (Size {:}) >> Inference time: {:f}".format(batch_idx, actual_batch_size, time.time() - start_time))

            class_predictions.extend(outputs[0][:actual_batch_size])
            batch_idx += 1

        class_list = self.class_list[:self.num_images]
        num_matches = np.sum(np.array(class_list) == np.array(class_predictions))
        accuracy = float(num_matches) / len(class_list)
        return accuracy
