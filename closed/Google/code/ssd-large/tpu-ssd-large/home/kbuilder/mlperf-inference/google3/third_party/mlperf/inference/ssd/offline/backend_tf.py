"""tensorflow backend (https://github.com/tensorflow/tensorflow)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

import functools
import tensorflow as tf

import backend
import convert_checkpoint
import packing_utils
import ssd_constants
import ssd_model
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu
from tensorflow.core.protobuf import rewriter_config_pb2  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import function  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import batch_ops  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu import functional as tpu_functional  # pylint: disable=g-direct-tensorflow-import


def _get_unpacked_image(params, image):
  """Returns the unpacked image."""

  if params["conv0_space_to_depth"]:
    image = tf.reshape(image, [
        -1, 3, (ssd_constants.IMAGE_SIZE + ssd_constants.PADDING_SIZE) //
        ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE,
        (ssd_constants.IMAGE_SIZE + ssd_constants.PADDING_SIZE) //
        ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE
    ])
    image = tf.slice(
        image, [0, 0, 0, 0],
        [-1, -1, ssd_constants.IMAGE_SIZE // 2, ssd_constants.IMAGE_SIZE // 2])
    image = packing_utils.unpack(image,
                                 ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE,
                                 image_format="NCHW")
    # Transpose from NCHW to NHWC.
    image = tf.transpose(image, [0, 2, 3, 1])
  else:
    image = tf.reshape(
        image,
        [ssd_constants.IMAGE_SIZE, ssd_constants.IMAGE_SIZE, 1, -1])
    image = packing_utils.unpack(image, 0, image_format="HWCN")
    # Transpose from HWCN to NHWC.
    image = tf.transpose(image, [3, 0, 1, 2])

  return tf.cast(image, tf.bfloat16 if params["use_bfloat16"] else tf.float32)


def model_fn(params, inputs):
  image, source_id, raw_shape = inputs
  mean = tf.constant(ssd_constants.MEAN, shape=[1, 1, 1, 3])
  std = tf.constant(ssd_constants.STD, shape=[1, 1, 1, 3])
  if params["conv0_space_to_depth"]:
    mean = tf.tile(mean, [1, 1, 1, 4])
    std = tf.tile(std, [1, 1, 1, 4])

  image = _get_unpacked_image(params, image)
  image /= 255.
  image = (image - tf.cast(mean, image.dtype)) / tf.cast(std, image.dtype)

  predictions = ssd_model.ssd_model_fn(
      images=image, source_id=source_id, raw_shape=raw_shape, params=params)
  return predictions


class BackendTensorflow(backend.Backend):

  def __init__(self):
    super(BackendTensorflow, self).__init__()

  def version(self):
    return tf.__version__ + "/" + tf.__git_version__

  def name(self):
    return "tensorflow"

  def image_format(self):
    # By default tensorflow uses NHWC (the cpu implementation only does NHWC)
    return "NCHW"

  def tpu_call(self, *args):
    image, source_id, raw_shape = args[0]
    image = tf.reshape(image, [-1])
    inputs = [[image, source_id, raw_shape]]

    @function.Defun(capture_resource_var_by_value=False)
    def tpu_subgraph():
      return tpu.rewrite(functools.partial(model_fn, self.params), inputs)

    return tpu_functional.TPUPartitionedCall(
        args=tpu_subgraph.captured_inputs,
        device_ordinal=tpu_ops.tpu_ordinal_selector(),
        Tout=[o.type for o in tpu_subgraph.definition.signature.output_arg],
        f=tpu_subgraph)

  def server_op(self, inputs_to_tpu, num_batch_threads, max_batch_size,
                batch_timeout_micros, allowed_batch_sizes,
                max_enqueued_batches):

    @batch_ops.batch_function(
        num_batch_threads=num_batch_threads,
        max_batch_size=max_batch_size,
        batch_timeout_micros=batch_timeout_micros,
        allowed_batch_sizes=allowed_batch_sizes,
        max_enqueued_batches=max_enqueued_batches)
    def batched_tpu_computation(*args):
      """Forms a batch TPU computation."""
      return self.tpu_call(args)

    return batched_tpu_computation(*inputs_to_tpu)

  def offline_op(self, inputs_to_tpu):
    return self.tpu_call(inputs_to_tpu)

  def load(self,
           model_path,
           model_output_dir,
           image_list_inmemory,
           params,
           batch_size=128,
           master="local",
           scenario="Offline",
           batch_timeout_micros=20 * 1000):
    if params["use_fused_bn"]:
      model_path = convert_checkpoint.convert_checkpoint(
          model_path, model_output_dir)
    tpu_graph = tf.Graph()
    tpu_config = tf.ConfigProto(
        operation_timeout_in_ms=600 * 1000,
        allow_soft_placement=True,
        graph_options=tf.GraphOptions(
            rewrite_options=rewriter_config_pb2.RewriterConfig(
                disable_meta_optimizer=True)),
        isolate_session_state=True)
    self.sess = tf.Session(master, graph=tpu_graph, config=tpu_config)
    self.params = params

    with tpu_graph.as_default():
      image_list = tf.constant(image_list_inmemory, dtype=tf.int32)
      if scenario == "Offline":
        self.indices = tf.placeholder(shape=(batch_size[-1]), dtype=tf.int32)
        self.source_id = tf.placeholder(shape=(batch_size[-1]), dtype=tf.int32)
        self.raw_shape = tf.placeholder(
            shape=(batch_size[-1], 3), dtype=tf.int32)
        image = tf.gather(image_list, self.indices, axis=0)
        if not params["conv0_space_to_depth"]:
          # Transpose from [N, C, H, W] to [H, W, C, N]
          image = tf.transpose(image, [2, 3, 1, 0])
        self.predict_op = self.offline_op(
            (image, self.source_id, self.raw_shape))
      else:
        self.indices = tf.placeholder(dtype=tf.int32)
        self.source_id = tf.placeholder(dtype=tf.int32)
        self.raw_shape = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        image = tf.gather(image_list, self.indices, axis=0)
        self.predict_op = self.server_op(
            [image, self.source_id, self.raw_shape],
            num_batch_threads=16,
            max_batch_size=batch_size[-1],
            batch_timeout_micros=batch_timeout_micros,
            allowed_batch_sizes=batch_size,
            max_enqueued_batches=10000)

      self.sess.run(tpu.initialize_system())
      for param in tf.trainable_variables():
        tf.logging.info("  %s, %s, %s" %
                        (param.name, str(param.get_shape()), param.op.device))

      # Checkpoint's variable name: https://internal/6714143388205056
      tf.compat.v1.train.init_from_checkpoint(model_path, {
          "ssd1200/": "ssd1200/",
      })
      self.sess.run(tf.initializers.global_variables())

    return self

  def predict(self, inputs):
    indices, source_id, raw_shape = inputs
    return self.sess.run(self.predict_op, {
        self.indices: indices,
        self.source_id: source_id,
        self.raw_shape: raw_shape
    })
