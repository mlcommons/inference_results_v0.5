# Copyright 2019 Google. All Rights Reserved.
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
# ==============================================================================
"""Data loader and preprocessing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import packing_utils
import ssd_constants
import tf_example_decoder


class SSDInputReader(object):
  """Input reader for dataset."""

  def __init__(self, file_pattern, transpose_input=True, space_to_depth=False):
    self._file_pattern = file_pattern
    self._transpose_input = transpose_input
    self._space_to_detph = space_to_depth

  def __call__(self):
    example_decoder = tf_example_decoder.TfExampleDecoder()

    def _parse_example(data):
      """Parse each tf record example."""
      with tf.name_scope('augmentation'):
        source_id = data['source_id']
        image = data['image']  # dtype uint8
        raw_shape = tf.shape(image)
        boxes = data['groundtruth_boxes']
        classes = tf.reshape(data['groundtruth_classes'], [-1, 1])

        # Only 80 of the 90 COCO classes are used.
        class_map = tf.convert_to_tensor(ssd_constants.CLASS_MAP)
        classes = tf.gather(class_map, classes)
        classes = tf.cast(classes, dtype=tf.float32)

        image = tf.image.resize_images(
            image, size=(ssd_constants.IMAGE_SIZE, ssd_constants.IMAGE_SIZE))

        def trim_and_pad(inp_tensor, dim_1):
          """Limit the number of boxes, and pad if necessary."""
          inp_tensor = inp_tensor[:ssd_constants.MAX_NUM_EVAL_BOXES]
          num_pad = ssd_constants.MAX_NUM_EVAL_BOXES - tf.shape(inp_tensor)[0]
          inp_tensor = tf.pad(inp_tensor, [[0, num_pad], [0, 0]])
          return tf.reshape(inp_tensor,
                            [ssd_constants.MAX_NUM_EVAL_BOXES, dim_1])

        boxes, classes = trim_and_pad(boxes, 4), trim_and_pad(classes, 1)

        sample = {
            ssd_constants.IMAGE: image,
            ssd_constants.BOXES: boxes,
            ssd_constants.CLASSES: classes,
            ssd_constants.SOURCE_ID: tf.string_to_number(source_id, tf.int32),
            ssd_constants.RAW_SHAPE: raw_shape,
        }

        return sample

    dataset = tf.data.Dataset.list_files(self._file_pattern, shuffle=False)

    # Prefetch data from files.
    def _prefetch_dataset(filename):
      dataset = tf.data.TFRecordDataset(filename).prefetch(1)
      return dataset
    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            _prefetch_dataset, cycle_length=32, sloppy=False))

    # Parse the fetched records to input tensors for model function.
    dataset = dataset.map(example_decoder.decode, num_parallel_calls=64)
    dataset = dataset.map(_parse_example, num_parallel_calls=64)

    def _preprocess(example):
      """Preprocess for each image."""
      image = example[ssd_constants.IMAGE]
      # Reference code:
      # https://github.com/mlperf/inference/blob/master/v0.5/classification_and_detection/python/dataset.py#L254
      # mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)
      # img_data = np.array(img.getdata(), dtype=np.float32)
      # (im_width, im_height) = img.size
      # img = img_data.reshape(im_height, im_width, 3)
      # img = img - mean

      if self._space_to_detph:
        image = tf.reshape(image, [
            ssd_constants.IMAGE_SIZE // ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE,
            ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE,
            ssd_constants.IMAGE_SIZE // ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE,
            ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE, 3
        ])
        image = tf.transpose(image, [0, 2, 1, 3, 4])
        image = tf.reshape(image, [
            ssd_constants.IMAGE_SIZE // ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE,
            ssd_constants.IMAGE_SIZE // ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE,
            3 * ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE *
            ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE
        ])
        img_pad = (ssd_constants.PADDING_SIZE //
                   ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE)
        paddings = tf.constant(
            [[0, img_pad], [0, img_pad], [0, 0]], tf.int32)
        image = tf.pad(image, paddings)
        image = packing_utils.pack(
            image,
            space_to_depth_block_size=ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE)
      else:
        image = packing_utils.pack(image, space_to_depth_block_size=0)

      if self._transpose_input:
        # Transpose to [C, H, W].
        image = tf.transpose(image, [2, 0, 1])

      example[ssd_constants.IMAGE] = image
      return example

    dataset = dataset.map(_preprocess, num_parallel_calls=64)

    return dataset
