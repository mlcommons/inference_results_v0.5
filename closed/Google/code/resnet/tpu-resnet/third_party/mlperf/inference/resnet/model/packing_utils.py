# Lint as: python3
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
"""Packing and unpacking utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _pack_rgb(image):
  """Cast image to uint8 and pack four values into a int32.

  The input `image` is in an RGB format, whose channel dimension is three. This
  function pads the channel dimension before packing.

  For example, an image with [1, 4, 3]
  [[[R, G, B], [R, G, B], [R, G, B], [R, G, B]]]
  -->
  pad to [1, 4, 4]
  [[[R, G, B, P], [R, G, B, P], [R, G, B, P], [R, G, B, P]]]
  `P` denotes padding.
  -->
  pack to [1, 4, 1]
  [[(RGBP), (RGBP), (RGBP), (RGBP)]]

  Args:
    image: a tensor of shape [height, width, channel].

  Returns:
    A image tensor that is packed into tf.int32 and has a shape of [height,
    width, 1]
  """
  h, w, _ = image.get_shape().as_list()
  if image.get_shape()[2] != 3:
    raise ValueError(
        'image.shape[2] is {:d}, but must be 3.'.format(image.get_shape()[2]))
  image = tf.pad(image, [[0, 0], [0, 0], [0, 1]])
  image = tf.cast(image, tf.uint8)
  image = tf.bitcast(image, tf.int32)
  image = tf.reshape(image, [h, w, -1])
  return image


def _pack_space_to_depth(image, space_to_depth_block_size):
  """Cast image to uint8 and pack four values into a int32.

  The input `image` uses space-to-depth transform; that is, the channel
  dimension is (space_to_depth_block_size**2) * channel dimension. This funciton
  supports only space_to_depth_block_size=2 and channel=3. Also, for performance
  on TPU, this function swaps the depth and channel dimesion.

  For example, an image with [1, 1, 12]
  [[[0R, 0G, 0B, 1R, 1G, 1B, 2R, 2G, 2B, 3R, 3G, 3B]]]
  numbers denote the space order and RGB denote the channel.
  -->
  swap depth and channel dimension.
  [[[0R, 1R, 2R, 3R, 0G, 1G, 2G, 3G, 0B, 1B, 2B, 3B]]]
  -->
  pack to [1, 1, 3]
  [[(R:0123), (G:0123), (B:0123)]]
  The R (or G or B) values of all locations are packed into a single tf.int32.

  Args:
    image: a tensor of shape [height, width, 3].
    space_to_depth_block_size: an `int`; only block size=2 is supported.

  Returns:
    A image tensor that is packed into tf.int32.
  """
  if space_to_depth_block_size != 2:
    raise ValueError('space-to-depth block size is {:d}, but must be 2.'.format(
        space_to_depth_block_size))
  if image.get_shape()[2] != 12:
    raise ValueError(
        'image.shape[2] is {:d}, but must be 12.'.format(image.get_shape()[2]))
  h, w, _ = image.get_shape().as_list()
  image = tf.reshape(image, [h, w, -1, 3])
  image = tf.transpose(image, [0, 1, 3, 2])
  image = tf.cast(image, tf.uint8)
  image = tf.bitcast(image, tf.int32)
  return image


def pack(image, space_to_depth_block_size):
  """Cast image to uint8 and pack four values into a int32.

  See respective functions for details.

  Args:
    image: a tensor of shape [height, width, channel].
    space_to_depth_block_size: an `int` for the space-to-depth block size.

  Returns:
    A image tensor that is packed into tf.int32.
  """
  if space_to_depth_block_size != 0:
    return _pack_space_to_depth(image, space_to_depth_block_size)
  else:
    return _pack_rgb(image)


def _get_constant(value, tile_kernel):
  return tf.reshape(tf.constant(value, tf.int32), tile_kernel)


def _get_channel_rank(image_format):
  """Returns the rank of the channel dimension."""
  if image_format not in ['HWCN', 'NHWC', 'NCHW']:
    raise ValueError('Unimplemented images format: {:s}.'.format(image_format))
  return image_format.find('C')


def _get_kernel(channel_rank, multiplier):
  """Creates a kernel specified by channel_rank and multiplier.

  The kernel tensor is a 4-element tensor, where all values are one except the
  channel_rank-th one, whose value is multiplier.

  For example, channel_rank=2, multiplier=3, this function returns [1, 1, 3, 1].

  Args:
    channel_rank: an `int`.
    multiplier: an `int`
  Returns
    A tensor of shape [4].
  """
  if channel_rank > 3:
    raise ValueError('channel_rank is {:d}, but must be <=3.'.format(
        channel_rank))

  number_of_ranks = 4
  iota = tf.range(number_of_ranks)
  return tf.where(
      tf.equal(iota, channel_rank), multiplier * tf.ones_like(iota),
      tf.ones_like(iota))


def _unpack_rgb(images, image_format):
  """Unpack images.

  This function assumes the `input` images is packed using _pack_rgb, which
  means the channel dimension is padded before packing. Also, the channel
  dimension is 1.

  A example that `images` is a tensor of [1, 1, 4, 1] (image_format=NHWC).
  [[[[(RGBP)], [(RGBP)], [(RGBP)], [(RGBP)]]]]
  elements in parenthesis is a single tf.in32 value, which packs four tf.uint8.
  -->
  tile to [1, 1, 4, 3]
  [[[(RGBP), (RGBP), (RGBP)],
    [(RGBP), (RGBP), (RGBP)],
    [(RGBP), (RGBP), (RGBP)]
    [(RGBP), (RGBP), (RGBP)]]]
  -->
  mask out paddings
  [[[R---, -G--, --B-],
    [R---, -G--, --B-],
    [R---, -G--, --B-]
    [R---, -G--, --B-]]]
  -->
  bit cast shift
  [[[R, G, B],
    [R, G, B],
    [R, G, B]
    [R, G, B]]]

  Args:
    images: a tensor of shape [height, width, channel, batch], [batch, height,
      width, channel], or [batch, channel, height, width].
    image_format: a `string` that specifies the image format.

  Returns:
    An unpacked tensor in tf.int32.
  """
  channel_rank = _get_channel_rank(image_format)
  channel_dim = images.get_shape()[channel_rank]
  if channel_dim != 1:
    raise ValueError(
        'images channel dimension is {:d}, but must be 1.'.format(channel_dim))

  tile_kernel = _get_kernel(channel_rank, 3)

  images = tf.tile(images, tile_kernel)
  images = tf.bitwise.bitwise_and(
      images,
      _get_constant([0xff, 0xff00, 0xff0000], tile_kernel))
  images = tf.bitwise.right_shift(images,
                                  _get_constant([0, 8, 16], tile_kernel))
  return images


def _unpack_space_to_depth(images, space_to_depth_block_size, image_format):
  """Unpack images.

  This function assumes the `input` images is packed using _pack_space_to_depth,
  which means it is packed along the space dimension and the channel dimension
  of `images` is 3.

  A example that `images` is a tensor of [1, 1, 1, 3] (image_format=NHWC).
  [[[(R:0123), (G:0123), (B:0123)]]]
  -->
  tile to [1, 1, 1, 12]
  [[[(R:0123), (G:0123), (B:0123),
     (R:0123), (G:0123), (B:0123),
     (R:0123), (G:0123), (B:0123),
     (R:0123), (G:0123), (B:0123)]]]
  -->
  bit cast shift.
  [[[(R:0123), (G:0123), (B:0123),
     (R:1230), (G:1230), (B:1230),
     (R:2301), (G:2301), (B:2301),
     (R:3012), (G:3012), (B:3012)]]]
  -->
  mask out values in other bytes.
  [[[R0, G0, B0,
     R1, G1, B1,
     R2, G2, B2,
     R3, G3, B3]]]

  Args:
    images: a tensor of shape [height, width, channel, batch], [batch, height,
      width, channel], or [batch, channel, height, width].
    space_to_depth_block_size: an `int`; only block size=2 is supported.
    image_format: a `string` that specifies the image format.

  Returns:
    An unpacked tensor in tf.int32.
  """
  if space_to_depth_block_size != 2:
    raise ValueError('space-to-depth block size is {:d}, but must be 2.'.format(
        space_to_depth_block_size))

  channel_rank = _get_channel_rank(image_format)
  channel_dim = images.get_shape()[channel_rank]
  if channel_dim != 3:
    raise ValueError(
        'images channel dimension is {:d}, but must be 3.'.format(channel_dim))

  tile_kernel = _get_kernel(channel_rank, 4)
  bitwise_kernel = _get_kernel(channel_rank, 12)

  images = tf.tile(images, tile_kernel)
  images = tf.bitwise.right_shift(
      images, _get_constant([0, 0, 0, 8, 8, 8, 16, 16, 16, 24, 24, 24],
                            bitwise_kernel))
  images = tf.bitwise.bitwise_and(
      images, _get_constant([0xff] * 12, bitwise_kernel))

  return images


def unpack(images, space_to_depth_block_size, image_format):
  """Unpack images.

  See respective functions for details.

  Args:
    images: a tensor of shape [height, width, channel, batch] or [batch, height,
      width, channel].
    space_to_depth_block_size: an `int` to indicate the space-to-depth size.
    image_format: a `string` one of ['NCHW', 'HWCN', 'NHWC'].

  Returns:
    An unpacked tensor in tf.int32.
  """
  if image_format not in ['NCHW', 'HWCN', 'NHWC']:
    raise ValueError(
        'images format is {:s}, but must be one of [NCHW, HWCN, NHWC].'.format(
            image_format))
  if space_to_depth_block_size != 0:
    return _unpack_space_to_depth(images, space_to_depth_block_size,
                                  image_format)
  else:
    return _unpack_rgb(images, image_format)
