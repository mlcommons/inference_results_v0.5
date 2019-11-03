# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Basic sequence-to-sequence model with dynamic RNN support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf

from tpu import beam_search_decoder
from tpu import decoder
from tpu import model_helper
from tpu.utils import misc_utils as utils

utils.check_tensorflow_version()

__all__ = ["BaseModel"]


def dropout(shape, dtype, keep_ratio):
  """Dropout helper function."""
  return tf.math.floor(tf.random.uniform(shape, dtype=dtype) +
                       keep_ratio) / keep_ratio


class Attention(tf.contrib.rnn.RNNCell):
  """Attention class that exposes both output and attention state."""

  def __init__(self, cell):
    super(Attention, self).__init__()
    self._cell = cell

  @property
  def wrapped_cell(self):
    return self._cell

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size + self._cell.state_size.attention

  def __call__(self, inputs, state, scope=None):
    out, new_state = self._cell(inputs, state)
    return tf.concat([out, new_state.attention], -1), new_state


class BaseModel(object):
  """Sequence-to-sequence base class.
  """

  def __init__(self, hparams, mode, features, reuse=tf.AUTO_REUSE):
    """Create the model.

    Args:
      hparams: Hyperparameter configurations.
      mode: TRAIN | EVAL | INFER
      features: a dict of input features.
      reuse: whether to reuse variables.
    """
    if reuse is not None:
      self.reuse = reuse
    else:
      self.reuse = mode == tf.contrib.learn.ModeKeys.INFER

    # Set params
    self._set_params_initializer(hparams, mode, features)

    # Train graph
    self.init_embeddings(hparams)
    source = features["source"]

    def f(seq_len):
      return lambda: self.build_graph(hparams, source, seq_len)

    def c(seq_len):
      return tf.reduce_max(
          features["source_sequence_length"]) < tf.constant(seq_len)

    res = [
        None,
        tf.cond(
            c(64), lambda: tf.cond(c(32), f(32), f(64)),
            lambda: tf.cond(c(96), f(96), f(128)))
    ]
    self._set_train_or_infer(res, hparams)

  def _emb_lookup(self, weight, index):
    return tf.cast(
        tf.reshape(
            tf.gather(weight, tf.reshape(index, [-1])),
            [index.shape[0], index.shape[1], -1]), self.dtype)

  def _set_params_initializer(self, hparams, mode, features):
    """Set various params for self and initialize."""
    self.mode = mode
    self.src_vocab_size = hparams.src_vocab_size
    self.tgt_vocab_size = hparams.tgt_vocab_size
    self.features = features

    self.dtype = tf.as_dtype(hparams.activation_dtype)

    self.single_cell_fn = None

    # Set num units
    self.num_units = hparams.num_units
    self.eos_id = hparams.tgt_eos_id
    self.label_smoothing = hparams.label_smoothing

    # Set num layers
    self.num_encoder_layers = hparams.num_encoder_layers
    self.num_decoder_layers = hparams.num_decoder_layers
    assert self.num_encoder_layers
    assert self.num_decoder_layers

    # Batch size
    self.batch_size = hparams.infer_batch_size

    # Global step
    # Use get_global_step instead of user-defied global steps. Otherwise the
    # num_train_steps in TPUEstimator.train has no effect (will train forever).
    # TPUestimator only check if tf.train.get_global_step() < num_train_steps
    self.global_step = None

    # Initializer
    self.random_seed = hparams.random_seed
    initializer = model_helper.get_initializer(
        hparams.init_op, self.random_seed, hparams.init_weight)
    tf.get_variable_scope().set_initializer(initializer)

  def _set_train_or_infer(self, res, hparams):
    """Set up training."""
    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.predicted_ids = res[1]

    params = tf.trainable_variables()

    # Gradients and SGD update operation for training the model.
    # Arrange for the embedding vars to appear at the beginning.
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      loss = res[0]
      self.learning_rate = tf.constant(hparams.learning_rate)
      # warm-up
      self.learning_rate = self._get_learning_rate_warmup(hparams)
      # decay
      self.learning_rate = self._get_learning_rate_decay(hparams)

      # Optimizer
      if hparams.optimizer == "sgd":
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      elif hparams.optimizer == "adam":
        opt = tf.train.AdamOptimizer(self.learning_rate)
      else:
        raise ValueError("Unknown optimizer type %s" % hparams.optimizer)

      opt = tf.contrib.tpu.CrossShardOptimizer(opt)
      # Gradients
      gradients = tf.gradients(loss, params, colocate_gradients_with_ops=True)

      clipped_grads, grad_norm = model_helper.gradient_clip(
          gradients, max_gradient_norm=hparams.max_gradient_norm)
      self.grad_norm = grad_norm

      self.update = opt.apply_gradients(
          zip(clipped_grads, params), global_step=self.global_step)

    # Print trainable variables
    utils.print_out("# Trainable variables")
    utils.print_out("Format: <name>, <shape>, <(soft) device placement>")
    for param in params:
      utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                        param.op.device))

  def _get_learning_rate_warmup(self, hparams):
    """Get learning rate warmup."""
    warmup_steps = hparams.warmup_steps
    warmup_scheme = hparams.warmup_scheme
    utils.print_out("  learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" %
                    (hparams.learning_rate, warmup_steps, warmup_scheme))

    # Apply inverse decay if global steps less than warmup steps.
    # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
    # When step < warmup_steps,
    #   learing_rate *= warmup_factor ** (warmup_steps - step)
    if warmup_scheme == "t2t":
      # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
      warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
      inv_decay = warmup_factor**(tf.to_float(warmup_steps - self.global_step))
    else:
      raise ValueError("Unknown warmup scheme %s" % warmup_scheme)

    return tf.cond(
        self.global_step < hparams.warmup_steps,
        lambda: inv_decay * self.learning_rate,
        lambda: self.learning_rate,
        name="learning_rate_warump_cond")

  def _get_learning_rate_decay(self, hparams):
    """Get learning rate decay."""
    return tf.cond(
        self.global_step < hparams.decay_start,
        lambda: self.learning_rate,
        lambda: tf.maximum(  # pylint: disable=g-long-lambda
            tf.train.exponential_decay(
                self.learning_rate,
                self.global_step - hparams.decay_start,
                hparams.decay_interval,
                hparams.decay_factor,
                staircase=True),
            self.learning_rate * tf.pow(hparams.decay_factor, hparams.
                                        decay_steps)),
        name="learning_rate_decay_cond")

  def init_embeddings(self, hparams):
    """Init embeddings."""
    self.embedding_encoder, self.embedding_decoder = (
        model_helper.create_emb_for_encoder_and_decoder(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            dtype=self.dtype,
            src_embed_size=self.num_units,
            tgt_embed_size=self.num_units,
            num_enc_partitions=hparams.num_enc_emb_partitions,
            num_dec_partitions=hparams.num_dec_emb_partitions,
            src_vocab_file=hparams.src_vocab_file,
            tgt_vocab_file=hparams.tgt_vocab_file,
            src_embed_file=hparams.src_embed_file,
            tgt_embed_file=hparams.tgt_embed_file,
        ))

  def build_graph(self, hparams, source, max_seq_len):
    """Subclass must implement this method.

    Creates a sequence-to-sequence model with dynamic RNN decoder API.
    Args:
      hparams: Hyperparameter configurations.
      source: The input source.
      max_seq_len: The max sequence length

    Returns:
      A tuple of the form (logits, predicted_ids) for infererence and
      (loss, None) for training.
      where:
        logits: float32 Tensor [batch_size x num_decoder_symbols]
        loss: float32 scalar
        predicted_ids: predicted ids from beam search.
    """
    utils.print_out("# Creating %s graph ..." % self.mode)

    source = tf.reshape(
        tf.slice(source, [0, 0], [self.batch_size, max_seq_len]),
        [self.batch_size, max_seq_len])
    with tf.variable_scope(
        "dynamic_seq2seq", dtype=self.dtype, reuse=self.reuse):
      if hparams.activation_dtype == "bfloat16":
        tf.get_variable_scope().set_dtype(tf.bfloat16)
      # Encoder
      encoder_outputs, encoder_states = self._build_encoder(hparams, source)

      ## Decoder
      with tf.variable_scope("decoder", reuse=self.reuse):
        with tf.variable_scope("output_projection", reuse=self.reuse):
          self.output_layer = tf.slice(
              tf.get_variable(
                  "kernel",
                  [self.num_units, 128 * (self.tgt_vocab_size // 128 + 1)]),
              [0, 0], [self.num_units, self.tgt_vocab_size])

      return self._build_decoder(encoder_outputs, encoder_states, hparams)[1]

  def _compute_loss(self, theta, inputs, factored_batch_size=None):
    """Final projection layer and computes the loss."""
    logits = tf.cast(
        tf.matmul(tf.cast(inputs[0], theta.dtype), theta), tf.float32)
    if factored_batch_size is not None:
      logits.set_shape([factored_batch_size, self.tgt_vocab_size])

    target = tf.cast(tf.reshape(inputs[1], [-1]), tf.int32)
    crossent = tf.losses.softmax_cross_entropy(
        tf.one_hot(target, self.tgt_vocab_size, dtype=logits.dtype),
        logits,
        label_smoothing=self.label_smoothing,
        reduction=tf.losses.Reduction.NONE)
    crossent = tf.where(target == self.eos_id, tf.zeros_like(crossent),
                        crossent)
    return tf.reshape(crossent, [-1]), []

  def _build_decoder(self, encoder_outputs, encoder_states, hparams):
    """Build and run a RNN decoder with a final projection layer.

    Args:
      encoder_outputs: The outputs of encoder for every time step.
      encoder_states: The encoder states.
      hparams: The Hyperparameters configurations.

    Returns:
      For inference, A tuple of final logits and final decoder state:
        logits: size [time, batch_size, vocab_size]
      For training, returns the final loss
    """
    with tf.variable_scope("decoder", reuse=self.reuse) as decoder_scope:
      memory = tf.transpose(encoder_outputs, [1, 0, 2])
      source_sequence_length = self.features["source_sequence_length"]
      if self.mode == tf.contrib.learn.ModeKeys.INFER:
        memory = tf.contrib.seq2seq.tile_batch(
            memory, multiplier=hparams.beam_width)
        source_sequence_length = tf.contrib.seq2seq.tile_batch(
            source_sequence_length, multiplier=hparams.beam_width)

      score_mask_value = tf.convert_to_tensor(
          tf.as_dtype(memory.dtype).as_numpy_dtype(-np.inf))
      attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
          hparams.num_units,
          memory,
          memory_sequence_length=source_sequence_length,
          score_mask_value=score_mask_value,
          normalize=True,
          dtype=memory.dtype)
      cell = tf.contrib.rnn.BasicLSTMCell(hparams.num_units, forget_bias=1.0)
      atten_cell = Attention(
          tf.contrib.seq2seq.AttentionWrapper(
              cell,
              attention_mechanism,
              attention_layer_size=None,
              alignment_history=False,
              output_attention=False,
              name="attention"))
      cells = []
      for i in range(3):
        with tf.variable_scope("uni_%d" % i, reuse=self.reuse):
          cells.append(
              tf.contrib.rnn.BasicLSTMCell(hparams.num_units, forget_bias=1.0))

      ## Train
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        target_input = self.features["target_input"]
        batch_size, _ = target_input.shape
        target_input = tf.transpose(target_input)

        emb = self._emb_lookup(self.embedding_decoder, target_input)
        seq_len = self.features["target_sequence_length"]
        out, _ = tf.contrib.recurrent.functional_rnn(
            atten_cell,
            emb * dropout(emb.shape, emb.dtype, 1.0 - hparams.dropout),
            dtype=self.dtype,
            sequence_length=seq_len,
            scope=decoder_scope,
            time_major=True,
            use_tpu=True)
        out, attention = tf.split(out, 2, -1)
        for i in range(3):
          with tf.variable_scope("uni_%d" % i, reuse=self.reuse) as s:
            inp = out
            out = tf.concat([out, attention], -1)
            out, _ = tf.contrib.recurrent.functional_rnn(
                cells[i],
                out * dropout(out.shape, emb.dtype, 1.0 - hparams.dropout),
                dtype=self.dtype,
                sequence_length=seq_len,
                scope=s,
                time_major=True,
                use_tpu=True)
            if i > 0:
              out += inp

        return tf.reduce_sum(
            self._compute_loss(self.output_layer, [
                tf.reshape(out, [-1, self.num_units]),
                tf.transpose(self.features["target_output"])
            ])[0]), None

      ## Inference
      else:
        assert hparams.infer_mode == "beam_search"
        start_tokens = tf.fill([self.batch_size], hparams.tgt_sos_id)
        end_token = hparams.tgt_eos_id
        beam_width = hparams.beam_width
        batch_size = self.batch_size * beam_width
        length_penalty_weight = hparams.length_penalty_weight
        coverage_penalty_weight = hparams.coverage_penalty_weight

        # maximum_iteration: The maximum decoding steps.
        maximum_iterations = hparams.tgt_max_len_infer

        def cell_fn(inputs, state):
          """Cell function used in decoder."""
          with tf.variable_scope(
              "multi_rnn_cell/cell_0_attention", reuse=self.reuse):
            o, s = atten_cell(inputs, state[0])
            o, attention = tf.split(o, 2, -1)
            new_state = [s]
          for i in range(3):
            with tf.variable_scope(
                "multi_rnn_cell/cell_%d" % (i + 1), reuse=self.reuse):
              inp = o
              o = tf.concat([o, attention], -1)
              o, s = cells[i](o, state[i + 1])
              new_state.append(s)
              if i > 0:
                o = inp + o
          return new_state, o

        encoder_states = [
            tf.contrib.seq2seq.tile_batch(i, beam_width) for i in encoder_states
        ]
        state0 = [
            atten_cell.zero_state(
                batch_size, self.dtype).clone(cell_state=encoder_states[0])
        ]
        for i in range(1, 4):
          state0.append(encoder_states[i])

        my_decoder = beam_search_decoder.BeamSearchDecoder(
            cell=cell_fn,
            embedding=self.embedding_decoder,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=state0,
            beam_width=beam_width,
            output_layer=self.output_layer,
            max_tgt=maximum_iterations,
            length_penalty_weight=length_penalty_weight,
            coverage_penalty_weight=coverage_penalty_weight,
            dtype=self.dtype)

        # Dynamic decoding
        predicted_ids = decoder.dynamic_decode(
            my_decoder,
            maximum_iterations=maximum_iterations,
            swap_memory=True,
            scope=decoder_scope)

    return None, predicted_ids

  def _build_encoder(self, hparams, source):
    """Build a GNMT encoder."""
    source = tf.transpose(source)

    with tf.variable_scope("encoder", reuse=self.reuse):
      emb = self._emb_lookup(self.embedding_encoder, source)
      seq_len = self.features["source_sequence_length"]
      if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
        emb = emb * dropout(emb.shape, emb.dtype, 1.0 - hparams.dropout)

      with tf.variable_scope("bi_fwd", reuse=self.reuse):
        fwd_cell = tf.contrib.rnn.BasicLSTMCell(
            hparams.num_units, reuse=self.reuse, forget_bias=1.0)
      with tf.variable_scope("bi_bwd", reuse=self.reuse):
        bwd_cell = tf.contrib.rnn.BasicLSTMCell(
            hparams.num_units, reuse=self.reuse, forget_bias=1.0)

      bi_outputs, bi_state = tf.contrib.recurrent.bidirectional_functional_rnn(
          fwd_cell,
          bwd_cell,
          emb,
          dtype=emb.dtype,
          sequence_length=seq_len,
          time_major=True,
          use_tpu=True)

      encoder_states = [bi_state[1]]

      out = tf.concat(bi_outputs, -1)

      for i in range(3):
        inp = out
        with tf.variable_scope(
            "rnn/multi_rnn_cell/cell_%d" % i, reuse=self.reuse) as scope:
          cell = tf.contrib.rnn.BasicLSTMCell(
              hparams.num_units, reuse=self.reuse, forget_bias=1.0)
          out, state = tf.contrib.recurrent.functional_rnn(
              cell,
              inp * dropout(inp.shape, emb.dtype, 1.0 - hparams.dropout)
              if self.mode == tf.contrib.learn.ModeKeys.TRAIN else inp,
              dtype=self.dtype,
              sequence_length=seq_len,
              time_major=True,
              scope=scope,
              use_tpu=True)
          encoder_states.append(state)
        if i > 0:
          out += inp
      return out, encoder_states
