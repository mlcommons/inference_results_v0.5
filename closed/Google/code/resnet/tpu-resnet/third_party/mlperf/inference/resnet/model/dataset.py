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
"""dataset related classes and methods."""

import ctypes
import sys
import time

import numpy as np
import tensorflow as tf

MEAN_RGB = [123.68, 116.78, 103.94]
IMAGE_SIZE = 224


class Item(object):

  def __init__(self, label, img, idx):
    self.label = label
    self.img = img
    self.idx = idx
    self.start = time.time()


def usleep(sec):
  """A cross-plattform sleep utility function."""
  if sys.platform == 'win32':
    # on windows time.sleep() doesn't work to well
    kernel32 = ctypes.windll.kernel32
    timer = kernel32.CreateWaitableTimerA(ctypes.c_void_p(), True,
                                          ctypes.c_void_p())
    delay = ctypes.c_longlong(int(-1 * (10 * 1000000 * sec)))
    kernel32.SetWaitableTimer(timer, ctypes.byref(delay), 0, ctypes.c_void_p(),
                              ctypes.c_void_p(), False)
    kernel32.WaitForSingleObject(timer, 0xffffffff)
  else:
    time.sleep(sec)


class Dataset(object):
  """Dataset class definition."""

  def __init__(self):
    self.arrival = None
    self.image_list = []
    self.label_list = []
    self.image_list_inmemory = []
    self.image_map = {}
    self.last_loaded = -1

  def preprocess(self, use_cache=True):
    raise NotImplementedError('Dataset:preprocess')

  def get_item_count(self):
    return len(self.image_list)

  def get_list(self):
    raise NotImplementedError('Dataset:get_list')

  def load_query_samples(self, sample_list):
    self.image_list_inmemory = []
    self.image_map = {}
    for i, sample in enumerate(sample_list):
      self.image_map[sample] = i
      img, _ = self.get_item(sample)
      self.image_list_inmemory.append(img)
    self.last_loaded = time.time()

  def unload_query_samples(self, sample_list):
    if sample_list:
      for sample in sample_list:
        del self.image_map[sample]
    else:
      self.image_map = {}
    self.image_list_inmemory = []

  def get_image_list_inmemory(self):
    return np.array(self.image_list_inmemory)

  def get_indices(self, item_list):
    data = [self.image_map[item] for item in item_list]
    data = np.array(data)
    return data, self.label_list[item_list]

  def get_item_loc(self, item):
    raise NotImplementedError('Dataset:get_item_loc')


#
# Post processing
#
class PostProcessCommon(object):
  """Class definition for common post-processing."""

  def __init__(self, offset=0):
    self.offset = offset
    self.good = 0
    self.total = 0

  def __call__(self, results, ids, expected=None, result_dict=None):
    n = len(results[0])
    for idx in range(0, n):
      if results[0][idx] + self.offset == expected[idx]:
        self.good += 1
    self.total += n
    return results

  def start(self):
    self.good = 0
    self.total = 0

  def finalize(self, results, unused_ds=False, unused_output_dir=None):
    results['good'] = self.good
    results['total'] = self.total


class PostProcessArgMax(object):
  """Class definition for argmax post-processing."""

  def __init__(self, offset=0):
    self.offset = offset
    self.good = 0
    self.total = 0

  def __call__(self, results, ids, expected=None, result_dict=None):
    result = np.argmax(results[0], axis=1)
    n = result.shape[0]
    for idx in range(0, n):
      if result[idx] + self.offset == expected[idx]:
        self.good += 1
    self.total += n
    return results

  def start(self):
    self.good = 0
    self.total = 0

  def finalize(self, results, unused_ds=False, unused_output_dir=None):
    results['good'] = self.good
    results['total'] = self.total


#
# pre-processing
#
def _resize_with_aspectratio(image_bytes,
                             out_height=IMAGE_SIZE,
                             out_width=IMAGE_SIZE,
                             scale=87.5):
  """Resize with aspect ratio."""

  # Reference: https://github.com/mlperf/inference/blob/master/v0.5/classification_and_detection/python/dataset.py#L154  # pylint: disable=line-too-long
  # Take width/height
  shape = tf.shape(image_bytes)
  height = shape[0]
  width = shape[1]

  new_height = 100.0 * tf.to_float(out_height) / scale
  new_width = 100.0 * tf.to_float(out_width) / scale
  h_greater_than_w = tf.greater(height, width)
  # Reference code:
  # if height > width:
  #      w = new_width
  #      h = int(height * new_width / width)
  #  else:
  #      h = new_height
  #      w = int(width * new_height / height)
  w = tf.to_int32(
      tf.cond(h_greater_than_w, lambda: new_width,
              lambda: tf.to_float(width) * new_height / tf.to_float(height)))
  h = tf.to_int32(
      tf.cond(h_greater_than_w,
              lambda: tf.to_float(height) * new_width / tf.to_float(width),
              lambda: new_height))

  image_bytes = tf.reshape(image_bytes, [1, height, width, 3])
  image_bytes = tf.image.resize_bilinear(image_bytes, (h, w))
  image_bytes = tf.reshape(image_bytes, [h, w, 3])
  return image_bytes


def _center_crop(image, size):
  """Crops to center of image with specified `size`."""

  # Reference: https://github.com/mlperf/inference/blob/master/v0.5/classification_and_detection/python/dataset.py#L144  # pylint: disable=line-too-long

  height = tf.shape(image)[0]
  width = tf.shape(image)[1]
  out_height = size
  out_width = size
  # Reference code:
  # left = (width - out_width) / 2
  # right = (width + out_width) / 2
  # top = (height - out_height) / 2
  # bottom = (height + out_height) / 2
  # img = img.crop((left, top, right, bottom))

  offset_height = tf.to_int32((height - out_height) / 2)
  offset_width = tf.to_int32((width - out_width) / 2)
  image = tf.image.crop_to_bounding_box(
      image,
      offset_height,
      offset_width,
      target_height=out_height,
      target_width=out_width,
  )
  return image


def _normalize(image):
  """Normalize the image."""
  offset = tf.constant(MEAN_RGB, shape=[1, 1, 3])
  image -= offset
  return image


def pre_process_vgg(image_bytes, label, image_name):
  """Preprocesses the given image for evaluation.

  Note that image normalization is moved to TPU. see backend_tf.py.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    label: label for that image.
    image_name: unique image identifier

  Returns:
    A preprocessed image `Tensor`.
  """
  image_size = IMAGE_SIZE
  image = _resize_with_aspectratio(image_bytes)
  image = _center_crop(image, image_size)
  image = tf.reshape(image, [image_size, image_size, 3])
  return image, label, image_name
