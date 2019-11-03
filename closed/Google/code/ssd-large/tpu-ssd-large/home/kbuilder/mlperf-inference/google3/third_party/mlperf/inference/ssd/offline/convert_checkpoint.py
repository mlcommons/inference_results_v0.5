r"""Tool to convert checkpoints.

The tool currently is used only to translate one checkpoint to another
whose batch normalization variables are fused into weight and bias of
previous conv layer.

To run the converter:
bazel build -c opt third_party/mlperf/inference/ssd/offline:convert_checkpoint &
convert_checkpoint \
    --checkpoint_file=<checkpoint to be converted> \
    --output_dir=<directory for converted checkpoint> \
    --gfs_user=tpu-perf-team --logtostderr
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf

import ssd_model

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_file", "", "Filename of the checkpoint.")
flags.DEFINE_string("output_dir", "", "Output dir.")

BATCH_NORM_EPSILON = 1e-5


def build_model_fn():
  """Builds model architecture."""

  # [N, H, W, C] with space-to-depth transformation.
  images = tf.placeholder(tf.float32, [1, 600, 600, 12])
  source_id = tf.placeholder(tf.int32, [1])
  raw_shape = tf.placeholder(tf.int32, [1, 3])

  params = dict(ssd_model.default_hparams().values())
  params["conv0_space_to_depth"] = True
  params["use_bfloat16"] = False
  params["use_fused_bn"] = True

  ssd_model.ssd_model_fn(
      images=images, source_id=source_id, raw_shape=raw_shape, params=params)


def get_checkpoint_reader(checkpoint_file):
  """Returns a CheckpointReader."""
  reader = tf.train.NewCheckpointReader(checkpoint_file)
  var_shapes = reader.get_variable_to_shape_map()
  var_dtypes = reader.get_variable_to_dtype_map()
  var_names = sorted(list(var_shapes.keys()))
  logging.info(
      "variables in checkpoint %s: name, shape, dtype", checkpoint_file)
  for v in var_names:
    logging.info("%s: %s %s", v, var_shapes[v], var_dtypes[v])
  return reader


def save_checkpoint(ops, output_checkpoint):
  """Saves the checkpoint after running the ops."""
  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(ops)
    save_path = saver.save(sess, output_checkpoint)
    logging.info("Model saved in path: %s", save_path)
    return save_path


def get_fused_weights(reader, weights_to_fuse, epsilon):
  """Fuses batch normalization into previous convolution layer.

    gamma * (x * kernel + bias) / sqrt(moving_variance + epsilon) +
    (beta - gamma * moving_mean / sqrt(moving_variance + epsilon))
  = x * (gamma * kernel / sqrt(moving_variance + epsilon)) +
    (beta + gamma * (bias - moving_mean) / sqrt(moving_variance + epsilon))

    As the original convolution layer does not have bias in MLPerf inference
    ResNet this particular case, the equation becomes:
    x * (kernel * gamma / sqrt(moving_variance + epsilon)) +
    beta - moving_mean * gamma / sqrt(moving_variance + epsilon)

  Args:
    reader: A CheckpointReader to obtain the tensor values for calculation.
    weights_to_fuse: A list of string, names of the variables that are going to
      be fused.
    epsilon: A small float, used for batch normalization to avoid dividing by 0.

  Returns:
    fused_kernel: A numpy 'ndarray', with the value of the fused conv kernel.
    fused_bias: A numpy 'ndarray', with the value of the fused conv bias.
  """
  kernel, beta, gamma, mean, variance = [
      reader.get_tensor(x) for x in weights_to_fuse]

  scaling_factor = gamma / np.sqrt(variance + epsilon)
  fused_kernel = kernel * scaling_factor
  fused_bias = beta - mean * scaling_factor

  return fused_kernel, fused_bias


def get_weights(reader):
  """Get the new weights."""
  new_vars = tf.global_variables()
  ops = []

  # No conversion for non-conv variables.
  for v in new_vars:
    var_name = v.name[:-2]  # Remove the ":0".
    if ("kernel" not in var_name and
        "bias" not in var_name or
        "additional_layers" in var_name or
        "multibox_head" in var_name):
      logging.info("No need for converting: %s.", var_name)
      if "bn" in var_name:
        logging.info(
            "Will not be used for inference after converting the checkpoint.")
      val = reader.get_tensor(var_name)
      op = v.assign(val)
      ops.append(op)

  # Fusing batch norm variables into previous conv. Currently hard-coded
  # based on MLPerf Inference SSD model.
  scopes_to_xform = [
      "ssd1200/conv1/conv1_1",
      "ssd1200/stage1_residul_block1/stage1_residul_block1_1",
      "ssd1200/stage1_residul_block1/stage1_residul_block1_2",
      "ssd1200/stage1_residul_block2/stage1_residul_block2_1",
      "ssd1200/stage1_residul_block2/stage1_residul_block2_2",
      "ssd1200/stage1_residul_block3/stage1_residul_block3_1",
      "ssd1200/stage1_residul_block3/stage1_residul_block3_2",
      "ssd1200/stage2_downsample/stage2_downsample_1",
      "ssd1200/stage2_residul_block1_1/stage2_residul_block1_1_1",
      "ssd1200/stage2_residul_block1_2/stage2_residul_block1_2_1",
      "ssd1200/stage2_residul_block2/stage2_residul_block2_1",
      "ssd1200/stage2_residul_block2/stage2_residul_block2_2",
      "ssd1200/stage2_residul_block3/stage2_residul_block3_1",
      "ssd1200/stage2_residul_block3/stage2_residul_block3_2",
      "ssd1200/stage2_residul_block4/stage2_residul_block4_1",
      "ssd1200/stage2_residul_block4/stage2_residul_block4_2",
      "ssd1200/stage3_downsample/stage3_downsample_1",
      "ssd1200/stage3_residul_block1/stage3_residul_block1_1",
      "ssd1200/stage3_residul_block1/stage3_residul_block1_2",
      "ssd1200/stage3_residul_block2/stage3_residul_block2_1",
      "ssd1200/stage3_residul_block2/stage3_residul_block2_2",
      "ssd1200/stage3_residul_block3/stage3_residul_block3_1",
      "ssd1200/stage3_residul_block3/stage3_residul_block3_2",
      "ssd1200/stage3_residul_block4/stage3_residul_block4_1",
      "ssd1200/stage3_residul_block4/stage3_residul_block4_2",
      "ssd1200/stage3_residul_block5/stage3_residul_block5_1",
      "ssd1200/stage3_residul_block5/stage3_residul_block5_2",
      "ssd1200/stage3_residul_block6/stage3_residul_block6_1",
      "ssd1200/stage3_residul_block6/stage3_residul_block6_2"
  ]

  for scope in scopes_to_xform:
    if scope.endswith("1"):
      weights_to_fuse = [
          scope + "/kernel",
          scope[:-1] + "bn1/beta",
          scope[:-1] + "bn1/gamma",
          scope[:-1] + "bn1/moving_mean",
          scope[:-1] + "bn1/moving_variance",
      ]
    else:
      weights_to_fuse = [
          scope + "/kernel",
          scope[:-1] + "bn2/beta",
          scope[:-1] + "bn2/gamma",
          scope[:-1] + "bn2/moving_mean",
          scope[:-1] + "bn2/moving_variance",
      ]

    xformed_kernel, xformed_bias = get_fused_weights(
        reader, weights_to_fuse, epsilon=BATCH_NORM_EPSILON)

    with tf.variable_scope(scope, reuse=True):
      var_kernel = tf.get_variable("kernel")
      var_bias = tf.get_variable("bias")

    ops.append(var_kernel.assign(xformed_kernel))
    ops.append(var_bias.assign(xformed_bias))

  return ops


def convert_checkpoint(checkpoint_file, output_dir):
  """Converts a given checkpoint into a new one.

  The function does the following steps:
  1. loads an existing checkpoint.
  2. computes the new variable values from the variables in the checkpoint.
  3. saves the new checkpoint in output_dir.

  Args:
    checkpoint_file: A string, filename of the checkpoint.
    output_dir: A string, output dir.

  Returns:
    The path to the converted checkpoint.
  """
  reader = get_checkpoint_reader(checkpoint_file)

  build_model_fn()
  ops = get_weights(reader)

  output_checkpoint = os.path.join(
      output_dir, os.path.basename(checkpoint_file))
  return save_checkpoint(ops, output_checkpoint)


def main(argv):
  del argv  # Unused.
  convert_checkpoint(FLAGS.checkpoint_file, FLAGS.output_dir)


if __name__ == "__main__":
  app.run(main)
