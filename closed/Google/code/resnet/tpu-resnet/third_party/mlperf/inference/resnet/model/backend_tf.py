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
"""TensorFlow backend (https://github.com/tensorflow/tensorflow)."""

import tensorflow as tf

from model import backend
from model import dataset
from model import packing_utils
from model import resnet_model
from tensorflow.core.protobuf import rewriter_config_pb2  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import function  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu import functional as tpu_functional  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu.ops import tpu_ops  # pylint: disable=g-direct-tensorflow-import


class BackendTensorflow(backend.Backend):
  """Class definition of TensorFlow backend."""

  def __init__(self, use_bfloat16=True, conv0_space_to_depth_block_size=2,
               tpu_transpose=False):
    self.use_bfloat16 = use_bfloat16
    self.conv0_space_to_depth_block_size = conv0_space_to_depth_block_size
    self.tpu_transpose = tpu_transpose
    self.tpu_config = tf.ConfigProto(
        operation_timeout_in_ms=600 * 1000,
        allow_soft_placement=True,
        graph_options=tf.GraphOptions(
            rewrite_options=rewriter_config_pb2.RewriterConfig(
                disable_meta_optimizer=True)),
        isolate_session_state=True)

  def version(self):
    return tf.__version__ + '/' + tf.__git_version__

  def name(self):
    return 'tensorflow'

  def image_format(self):
    # By default tensorflow uses NHWC (the cpu implementation only does NHWC).
    return 'NHWC'

  def tpu_call(self, *args):

    def model_fn(images):
      """model_fn for Resnet."""

      def build_network(images):
        """Builds the ResNet network architecture."""
        network = resnet_model.resnet_v1(
            resnet_depth=50,
            num_classes=1000,
            data_format='channels_last',
            conv0_kernel_size=7,
            conv0_space_to_depth_block_size=self.conv0_space_to_depth_block_size
        )

        logits = network(inputs=images, is_training=False)
        return logits

      # The followins shapes are w.r.t. byte packing. see packing_utils.py.
      if self.conv0_space_to_depth_block_size != 0:
        if self.tpu_transpose:
          images = tf.reshape(images, [112, 112, 3, -1])
        else:
          images = tf.reshape(images, [-1, 3, 112, 112])
      else:
        if self.tpu_transpose:
          images = tf.reshape(images, [224, 224, 1, -1])
        else:
          images = tf.reshape(images, [-1, 1, 224, 224])
      images = packing_utils.unpack(
          images,
          self.conv0_space_to_depth_block_size,
          image_format='HWCN' if self.tpu_transpose else 'NCHW')

      if self.tpu_transpose:
        # Transpose from [H, W, C, N] to [N, H, W, C]
        images = tf.transpose(images, [3, 0, 1, 2])
      else:
        # Transpose from [N, C, H, W] to [N, H, W, C]
        images = tf.transpose(images, [0, 2, 3, 1])

      def _normalize(images):
        """Normalize the images."""
        _, _, _, c = images.get_shape().as_list()
        offset = tf.constant(dataset.MEAN_RGB * (c // 3), shape=[1, 1, 1, c],
                             dtype=images.dtype)
        images -= offset
        return images

      images = tf.cast(images, tf.bfloat16 if self.use_bfloat16 else tf.float32)
      images = _normalize(images)

      if self.use_bfloat16:
        with tf.contrib.tpu.bfloat16_scope():
          logits = build_network(images)
        logits = tf.cast(logits, tf.float32)
      else:
        logits = build_network(images)
      return tf.argmax(logits, axis=1) - 1

    @function.Defun(capture_resource_var_by_value=False)
    def tpu_subgraph():
      return tf.tpu.rewrite(model_fn, args)

    return tpu_functional.TPUPartitionedCall(
        args=tpu_subgraph.captured_inputs,
        device_ordinal=tpu_ops.tpu_ordinal_selector(),
        Tout=[o.type for o in tpu_subgraph.definition.signature.output_arg],
        f=tpu_subgraph)

  def build_and_export(self,
                       model_path,
                       export_model_path,
                       batch_size=None,
                       master='local',
                       scenario='Offline'):
    export_graph = tf.Graph()
    export_sess = tf.Session(master, graph=export_graph, config=self.tpu_config)
    builder = tf.compat.v1.saved_model.Builder(export_model_path)
    with export_graph.as_default():
      next_image_list = tf.Variable(
          0, validate_shape=False, shape=tf.TensorShape(None), dtype=tf.int32)
      image_list = tf.Variable(
          0, validate_shape=False, shape=tf.TensorShape(None), dtype=tf.int32)
      assign = tf.assign(image_list, next_image_list, validate_shape=False)

      if scenario == 'Offline':
        indices = tf.placeholder(shape=(batch_size[-1]), dtype=tf.int32)
      else:
        indices = tf.placeholder(dtype=tf.int32)

      images = tf.gather(image_list, indices, axis=0)
      if self.tpu_transpose:
        # Transpose from [N, C, H, W] to [H, W, C, N]
        images = tf.transpose(images, [2, 3, 1, 0])

      self.predict_op = tf.cast(
          self.tpu_call(tf.reshape(images, [-1]))[0], dtype=tf.int32)

      export_sess.run(tf.tpu.initialize_system())
      tf.train.init_from_checkpoint(model_path, {
          'resnet_model/': '/',
      })
      export_sess.run(tf.initializers.global_variables())
      signature_def_map = {
          tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              tf.saved_model.predict_signature_def(
                  inputs={
                      'indices': indices,
                      'image_list': next_image_list,
                  },
                  outputs={
                      'logits': self.predict_op,
                      'assign': assign
                  })
      }
      export_sess.run(tf.tpu.shutdown_system())
      builder.add_meta_graph_and_variables(
          export_sess,
          tags=[
              tf.saved_model.tag_constants.SERVING,
              tf.saved_model.tag_constants.TPU
          ],
          signature_def_map=signature_def_map)
      tf.logging.info('graph saved.')
    builder.save()

  def load(self, export_model_path, master='local'):
    # Load the frozen graph for prediction.
    tpu_graph = tf.Graph()
    self.sess = tf.Session(master, graph=tpu_graph, config=self.tpu_config)
    with tpu_graph.as_default():

      meta_graph_def = tf.saved_model.load(
          self.sess,
          tags=[
              tf.saved_model.tag_constants.SERVING,
              tf.saved_model.tag_constants.TPU
          ],
          export_dir=export_model_path)
      for signature_def_key in sorted(meta_graph_def.signature_def.keys()):
        tf.logging.info('SignatureDef key: \"%s\"' % signature_def_key)
        # Get inputs
        self.inputs_tensor_info = meta_graph_def.signature_def[
            signature_def_key].inputs
        tf.logging.info(self.inputs_tensor_info)
        # Get outputs
        self.outputs_tensor_info = meta_graph_def.signature_def[
            signature_def_key].outputs
        tf.logging.info(self.outputs_tensor_info)
      self.sess.run(tf.tpu.initialize_system())

  def update_qsl(self, image_list):
    self.sess.run(
        [self.outputs_tensor_info['assign'].name],
        feed_dict={self.inputs_tensor_info['image_list'].name: image_list})

  def predict(self, indices):
    return self.sess.run(
        [self.outputs_tensor_info['logits'].name],
        feed_dict={self.inputs_tensor_info['indices'].name: indices})
