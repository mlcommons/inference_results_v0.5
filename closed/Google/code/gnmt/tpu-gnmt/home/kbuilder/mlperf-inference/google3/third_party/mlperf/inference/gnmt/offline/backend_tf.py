"""tensorflow backend (https://github.com/tensorflow/tensorflow)."""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation
from __future__ import print_function

import bisect
import functools
import numpy as np
import tensorflow as tf

from tpu import model
from tpu.utils import vocab_utils
import backend
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu
from tensorflow.core.protobuf import rewriter_config_pb2  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import function  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import batch_ops  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu import functional as tpu_functional  # pylint: disable=g-direct-tensorflow-import


def model_fn(hparams, inputs):
  source, source_sequence_length = inputs
  features = {}
  end_with_dot = tf.reshape(
      tf.equal(
          tf.gather(
              source,
              tf.expand_dims(source_sequence_length - 1, 1),
              batch_dims=1), 4), [-1])
  adding_tensor = tf.one_hot(
      source_sequence_length, source.shape[1], dtype=tf.int32,
      on_value=4) * (1 - tf.cast(tf.expand_dims(end_with_dot, 1), tf.int32))
  source = tf.math.maximum(source, adding_tensor)
  source_sequence_length = tf.where(end_with_dot, source_sequence_length,
                                    source_sequence_length + 1)
  features['source'] = source
  features['source_sequence_length'] = source_sequence_length

  # Create a GNMT model for inference.
  gnmt_model = model.BaseModel(
      hparams,
      mode=tf.contrib.learn.ModeKeys.INFER,
      features=features,
      reuse=tf.AUTO_REUSE)
  predicted_ids = tf.reshape(gnmt_model.predicted_ids,
                             [hparams.infer_batch_size, hparams.beam_width, -1])
  # make sure outputs is of shape [batch_size, time] or [beam_width,
  # batch_size, time] when using beam search.
  predicted_ids = tf.transpose(predicted_ids, [1, 0, 2])
  # Get the top predictions from beam search.
  predicted_ids = tf.gather_nd(predicted_ids, [0])
  return predicted_ids


class BackendTensorflow(backend.Backend):

  def __init__(self, vocab_prefix, scenario='Offline'):
    super(BackendTensorflow, self).__init__()
    self.vocab_prefix = vocab_prefix
    self.scenario = scenario
    self.predict_ops = []

  def version(self):
    return tf.__version__ + '/' + tf.__git_version__

  def name(self):
    return 'tensorflow'

  def tpu_call(self, args):

    @function.Defun(capture_resource_var_by_value=False)
    def tpu_subgraph():
      results = tpu.rewrite(functools.partial(model_fn, self.hparams), args)
      results = tf.reshape(results, [self.hparams.infer_batch_size, -1])
      return self.vocab_table.lookup(tf.to_int64(results))

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
      for arg in args:
        print('arg: ', arg)
      return self.tpu_call([args])

    return batched_tpu_computation(*inputs_to_tpu)

  def offline_op(self, inputs_to_tpu):
    return self.tpu_call(inputs_to_tpu)

  def load(self,
           ckpt_path,
           hparams,
           master='local',
           batch_timeout_micros=80 * 1000,
           buckets=None):
    self.hparams = hparams
    self.buckets = buckets
    self.tpu_graph = tf.Graph()
    tpu_config = tf.ConfigProto(
        operation_timeout_in_ms=600 * 1000,
        allow_soft_placement=True,
        graph_options=tf.GraphOptions(
            rewrite_options=rewriter_config_pb2.RewriterConfig(
                disable_meta_optimizer=True)),
        isolate_session_state=True)
    # Find tpu master.
    print('master value set to:', master)
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        master, zone=None, project=None)
    master = tpu_cluster_resolver.get_master()
    self.sess = tf.Session(master, graph=self.tpu_graph, config=tpu_config)
    with self.tpu_graph.as_default():
      self.vocab_table = tf.contrib.lookup.index_to_string_table_from_file(
          self.vocab_prefix, default_value=vocab_utils.UNK)

    if self.scenario == 'Offline':
      with self.tpu_graph.as_default():
        self.source = tf.placeholder(
            shape=(hparams.infer_batch_size, hparams.src_max_len_infer),
            dtype=tf.int32)
        self.source_sequence_length = tf.placeholder(
            shape=(hparams.infer_batch_size), dtype=tf.int32)

        inputs = [[self.source, self.source_sequence_length]]
        self.predict_ops.append(self.offline_op(inputs))
    else:
      with self.tpu_graph.as_default():
        self.source = tf.placeholder(
            shape=[None, hparams.src_max_len_infer], dtype=tf.int32)
        self.source_sequence_length = tf.placeholder(
            shape=[None], dtype=tf.int32)
        inputs = [self.source, self.source_sequence_length]
        for _ in buckets:
          self.predict_ops.append(self.server_op(
              inputs,
              num_batch_threads=16,
              max_batch_size=hparams.infer_batch_size,
              batch_timeout_micros=batch_timeout_micros,
              allowed_batch_sizes=[hparams.infer_batch_size],
              max_enqueued_batches=10000))
        # Add longest sequence predict op.
        self.predict_ops.append(self.server_op(
            inputs,
            num_batch_threads=16,
            max_batch_size=hparams.infer_batch_size,
            batch_timeout_micros=5000*1000,
            allowed_batch_sizes=[hparams.infer_batch_size],
            max_enqueued_batches=10000))

    with self.tpu_graph.as_default():
      vs = tf.global_variables()

      assign_ops = []
      var_map = {}
      with tf.variable_scope('f32', dtype=tf.float32):
        for i in vs:
          if 'output_projection' in i.name:
            new_var = tf.get_variable(i.name[:-2],
                                      [i.shape[0], hparams.tgt_vocab_size])
            assign_ops.append(
                tf.assign(
                    i,
                    tf.pad(
                        tf.cast(new_var, i.dtype),
                        [[0, 0],
                         [
                             0, 128 * (hparams.tgt_vocab_size // 128 + 1) -
                             hparams.tgt_vocab_size
                         ]])))
          else:
            new_var = tf.get_variable(i.name[:-2], i.shape)
            assign_ops.append(tf.assign(i, tf.cast(new_var, i.dtype)))
          var_map[i.name[:-2]] = new_var.name[:-2]

      self.sess.run(tpu.initialize_system())
      tf.train.init_from_checkpoint(ckpt_path, var_map)
      self.sess.run(tf.initializers.global_variables())
      self.sess.run(tf.tables_initializer())
      self.sess.run(assign_ops)

    return self

  def warmup(self):
    source = np.ones((self.hparams.infer_batch_size,
                      self.hparams.src_max_len_infer))
    source_sequence_length = np.ones((self.hparams.infer_batch_size)) * 10
    for predict_op in self.predict_ops:
      for _ in range(32):
        self.sess.run(predict_op,
                      {self.source: source,
                       self.source_sequence_length: source_sequence_length})

  def predict(self, source, source_sequence_length):
    if self.scenario == 'Offline':
      return self.sess.run(self.predict_ops[0], {
          self.source: source,
          self.source_sequence_length: source_sequence_length
      })
    else:
      max_seq = max(source_sequence_length)
      index = bisect.bisect_left(self.buckets, max_seq)
      return self.sess.run(self.predict_ops[index], {
          self.source: source,
          self.source_sequence_length: source_sequence_length
      })
