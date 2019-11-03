# python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""The implementation of imagenet dataset."""

from __future__ import print_function

import functools
import logging
import os
import time

import numpy as np
import tensorflow as tf

from model import dataset
from model import packing_utils

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('imagenet')


def dataset_parser(value):
  """Parses an image and its label from a serialized ResNet-50 TFExample.

  Reference:
  mlperf/models/rough/resnet/imagenet_input.py?l=168&rcl=248201270

  Args:
    value: serialized string containing an ImageNet TFExample.

  Returns:
    Returns a tuple of (image, label) from the TFExample.
  """
  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, ''),
      'image/format': tf.FixedLenFeature((), tf.string, 'jpeg'),
      'image/class/label': tf.FixedLenFeature([], tf.int64, -1),
      'image/class/text': tf.FixedLenFeature([], tf.string, ''),
      'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
      'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
      'image/filename': tf.FixedLenFeature([], tf.string, ''),
  }

  parsed = tf.parse_single_example(value, keys_to_features)
  image_bytes = tf.reshape(parsed['image/encoded'], shape=[])
  image_bytes = tf.io.decode_jpeg(image_bytes, 3)

  # Subtract one so that labels are in [0, 1000).
  label = tf.cast(
      tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32) - 1
  image_name = parsed['image/filename']
  return image_bytes, label, image_name


def image_transform(conv0_space_to_depth_block_size, need_transpose,
                    image, label, image_name):
  """Transforms images."""
  if conv0_space_to_depth_block_size != 0:
    height, width, features = image.get_shape().as_list()
    image = tf.reshape(
        image,
        [height // conv0_space_to_depth_block_size,
         conv0_space_to_depth_block_size,
         width // conv0_space_to_depth_block_size,
         conv0_space_to_depth_block_size, features])
    image = tf.transpose(image, [0, 2, 1, 3, 4])
    image = tf.reshape(
        image,
        [height // conv0_space_to_depth_block_size,
         width // conv0_space_to_depth_block_size,
         features * (conv0_space_to_depth_block_size ** 2)])

  image = packing_utils.pack(image, conv0_space_to_depth_block_size)

  if need_transpose:
    image = tf.transpose(image, [2, 0, 1])
  return image, label, image_name


class Imagenet(dataset.Dataset):
  """Imagenet dataset definition."""

  def __init__(self,
               data_path,
               image_list,
               name,
               use_cache=0,
               image_size=None,
               image_format='NHWC',
               need_transpose=True,
               conv0_space_to_depth_block_size=2,
               pre_process=None,
               count=None,
               cache_dir=None):
    super(Imagenet, self).__init__()
    if image_size is None:
      self.image_size = [224, 224, 3]
    else:
      self.image_size = image_size
    if not cache_dir:
      cache_dir = os.getcwd()
    self.image_list = []
    self.label_list = []
    self.count = count
    self.use_cache = use_cache
    self.cache_dir = os.path.join(cache_dir, 'preprocessed', name, image_format)
    self.data_path = data_path
    self.pre_process = pre_process
    # input images are in HWC
    self.need_transpose = need_transpose
    self.conv0_space_to_depth_block_size = conv0_space_to_depth_block_size

    not_found = 0
    tf.gfile.MakeDirs(self.cache_dir)

    start = time.time()
    filepattern = os.path.join(self.data_path, 'validation-*')
    data_set = tf.data.Dataset.list_files(filepattern, shuffle=False)

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024  # 8 MiB per file
      data_set = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return data_set

    # Read the data from disk in parallel
    data_set = data_set.apply(
        tf.contrib.data.parallel_interleave(
            fetch_dataset, cycle_length=8, sloppy=False))
    data_set = data_set.map(dataset_parser, num_parallel_calls=64)
    data_set = data_set.map(dataset.pre_process_vgg, num_parallel_calls=64)

    data_set = data_set.map(
        functools.partial(image_transform,
                          self.conv0_space_to_depth_block_size,
                          self.need_transpose),
        num_parallel_calls=64)
    iterator = data_set.make_initializable_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
      sess.run(iterator.initializer)
      while self.count == 0 or len(self.image_list) < self.count:
        try:
          processed, label, image_name = sess.run(next_element)
        except tf.errors.OutOfRangeError:
          break
        image_name_str = image_name.decode('utf-8')
        # All of the image_name look like:
        # ILSVRC2012_val_00042559.JPEG
        # image_name_str[15:23] extracts 00042559.
        image_id = int(image_name_str[15:23])
        dst_file = '%d.npy' % image_id
        dst = os.path.join(self.cache_dir, dst_file)
        with tf.gfile.Open(dst, 'wb') as fout:
          np.save(fout, processed)
        self.image_list.append(dst_file)
        self.label_list.append(int(label))

    time_taken = time.time() - start
    if not self.image_list:
      log.error('no images in image list found')
      raise ValueError('no images in image list found')
    if not_found > 0:
      log.info('reduced image list, %d images not found', not_found)

    log.info('loaded %d images, cache=%s, took=%1fsec', len(self.image_list),
             use_cache, time_taken)

    self.label_list = np.array(self.label_list)

  def get_item(self, nr):
    """Get image by number in the list."""
    dst = os.path.join(self.cache_dir, self.image_list[nr])
    with tf.gfile.Open(dst, 'rb') as fout:
      img = np.load(fout)
    return img, self.label_list[nr]

  def get_item_loc(self, nr):
    src = os.path.join(self.data_path, self.image_list[nr])
    return src
