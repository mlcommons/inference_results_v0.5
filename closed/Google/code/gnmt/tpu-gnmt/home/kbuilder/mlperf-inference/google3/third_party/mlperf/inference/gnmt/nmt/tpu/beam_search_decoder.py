# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""A decoder that performs beam search."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf

from tpu import decoder

__all__ = [
    "BeamSearchDecoderState",
    "BeamSearchDecoder",
]


class BeamSearchDecoderState(
    collections.namedtuple("BeamSearchDecoderState",
                           ("cell_state", "log_probs", "finished", "lengths",
                            "accumulated_attention_probs", "pred_ids"))):
  pass


class BeamSearchDecoder(decoder.Decoder):
  """BeamSearch sampling decoder.

    **NOTE** If you are using the `BeamSearchDecoder` with a cell wrapped in
    `AttentionWrapper`, then you must ensure that:

    - The encoder output has been tiled to `beam_width` via
      `tf.contrib.seq2seq.tile_batch` (NOT `tf.tile`).
    - The `batch_size` argument passed to the `zero_state` method of this
      wrapper is equal to `true_batch_size * beam_width`.
    - The initial state created with `zero_state` above contains a
      `cell_state` value containing properly tiled final state from the
      encoder.

    An example:

    ```
    tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
        encoder_outputs, multiplier=beam_width)
    tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
        encoder_final_state, multiplier=beam_width)
    tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
        sequence_length, multiplier=beam_width)
    attention_mechanism = MyFavoriteAttentionMechanism(
        num_units=attention_depth,
        memory=tiled_inputs,
        memory_sequence_length=tiled_sequence_length)
    attention_cell = AttentionWrapper(cell, attention_mechanism, ...)
    decoder_initial_state = attention_cell.zero_state(
        dtype, batch_size=true_batch_size * beam_width)
    decoder_initial_state = decoder_initial_state.clone(
        cell_state=tiled_encoder_final_state)
    ```

    Meanwhile, with `AttentionWrapper`, coverage penalty is suggested to use
    when computing scores(https://arxiv.org/pdf/1609.08144.pdf). It encourages
    the translation to cover all inputs.
  """

  def __init__(self,
               cell,
               embedding,
               start_tokens,
               end_token,
               initial_state,
               beam_width,
               output_layer=None,
               max_tgt=None,
               length_penalty_weight=0.0,
               coverage_penalty_weight=0.0,
               reorder_tensor_arrays=True,
               dtype=tf.float32):
    """Initialize the BeamSearchDecoder.

    Args:
      cell: An `RNNCell` instance.
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`.
      start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
      end_token: `int32` scalar, the token that marks end of decoding.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
      beam_width:  Python integer, the number of beams.
      output_layer: (Optional) An instance of `tf.variable` to represent the
        weights for the dense layer to apply to the RNN output prior
        to storing the result or sampling.
      max_tgt: maximum prediction length.
      length_penalty_weight: Float weight to penalize length. Disabled with 0.0.
      coverage_penalty_weight: Float weight to penalize the coverage of source
        sentence. Disabled with 0.0.
      reorder_tensor_arrays: If `True`, `TensorArray`s' elements within the cell
        state will be reordered according to the beam search path. If the
        `TensorArray` can be reordered, the stacked form will be returned.
        Otherwise, the `TensorArray` will be returned as is. Set this flag to
        `False` if the cell state contains `TensorArray`s that are not amenable
        to reordering.

    Raises:
      TypeError: if `cell` is not an instance of `RNNCell`,
        or `output_layer` is not an instance of `tf.layers.Layer`.
      ValueError: If `start_tokens` is not a vector or
        `end_token` is not a scalar.
    """
    self._cell = cell
    self._output_layer = output_layer
    self._reorder_tensor_arrays = reorder_tensor_arrays

    if callable(embedding):
      self._embedding_fn = embedding
    else:
      self._embedding_fn = (
          lambda ids: tf.cast(tf.nn.embedding_lookup(embedding, ids), dtype))

    self._start_tokens = tf.convert_to_tensor(
        start_tokens, dtype=tf.int32, name="start_tokens")
    if self._start_tokens.get_shape().ndims != 1:
      raise ValueError("start_tokens must be a vector")
    self._end_token = tf.convert_to_tensor(
        end_token, dtype=tf.int32, name="end_token")
    if self._end_token.get_shape().ndims != 0:
      raise ValueError("end_token must be a scalar")

    self._batch_size = start_tokens.shape[0].value
    self._beam_width = beam_width
    self._max_tgt = max_tgt
    self._length_penalty_weight = length_penalty_weight
    self._coverage_penalty_weight = coverage_penalty_weight
    self._initial_cell_state = initial_state
    self._start_tokens = tf.reshape(
        tf.tile(tf.expand_dims(self._start_tokens, 1), [1, self._beam_width]),
        [-1])

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def tracks_own_finished(self):
    """The BeamSearchDecoder shuffles its beams and their finished state.

    For this reason, it conflicts with the `dynamic_decode` function's
    tracking of finished states.  Setting this property to true avoids
    early stopping of decoding due to mismanagement of the finished state
    in `dynamic_decode`.

    Returns:
      `True`.
    """
    return True

  @property
  def output_size(self):
    return tf.TensorShape([self._batch_size * self._beam_width])

  @property
  def output_dtype(self):
    return tf.int32

  def initialize(self, name=None):
    """Initialize the decoder.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, start_inputs, initial_state)`.
    """
    finished = tf.constant(
        False, shape=[self._batch_size * self._beam_width], dtype=tf.bool)
    start_inputs = self._embedding_fn(self._start_tokens)

    dtype = tf.contrib.framework.nest.flatten(self._initial_cell_state)[0].dtype
    log_probs = tf.one_hot(  # shape(batch_sz, beam_sz)
        tf.zeros([self._batch_size], dtype=tf.int32),
        depth=self._beam_width,
        on_value=tf.convert_to_tensor(0.0, dtype=dtype),
        off_value=tf.convert_to_tensor(-np.Inf, dtype=dtype),
        dtype=dtype)
    init_attention_probs = get_attention_probs(
        self._initial_cell_state, self._coverage_penalty_weight)
    if init_attention_probs is None:
      init_attention_probs = ()
    init_pred_ids = tf.fill(
        [self._batch_size * self._beam_width, self._max_tgt], self._end_token)

    initial_state = BeamSearchDecoderState(
        cell_state=self._initial_cell_state,
        log_probs=tf.reshape(log_probs, [-1]),
        finished=finished,
        lengths=tf.zeros([self._batch_size * self._beam_width], dtype=tf.int32),
        accumulated_attention_probs=init_attention_probs,
        pred_ids=init_pred_ids)

    return (finished, start_inputs, initial_state)

  def step(self, time, inputs, state, name=None):
    """Perform a decoding step.

    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    batch_size = self._batch_size
    beam_width = self._beam_width
    end_token = self._end_token
    length_penalty_weight = self._length_penalty_weight
    coverage_penalty_weight = self._coverage_penalty_weight

    with tf.name_scope(name, "BeamSearchDecoderStep", (time, inputs, state)):
      cell_state = state.cell_state
      next_cell_state, cell_outputs = self._cell(inputs, cell_state)

      if self._output_layer is not None:
        output_shape = cell_outputs.shape.as_list()[:-1] + [
            self._output_layer.shape[-1]
        ]
        cell_outputs = tf.matmul(
            tf.reshape(cell_outputs, [-1, cell_outputs.shape.as_list()[-1]]),
            tf.cast(self._output_layer, cell_outputs.dtype))
        cell_outputs = tf.reshape(cell_outputs, output_shape)

      beam_search_output, beam_search_state = _beam_search_step(
          time=time,
          logits=cell_outputs,
          next_cell_state=next_cell_state,
          beam_state=state,
          batch_size=batch_size,
          beam_width=beam_width,
          end_token=end_token,
          length_penalty_weight=length_penalty_weight,
          coverage_penalty_weight=coverage_penalty_weight,
          max_tgt=self._max_tgt)

      finished = beam_search_state.finished
      sample_ids = tf.reshape(beam_search_output, [-1])
      next_inputs = self._embedding_fn(sample_ids)

    return (beam_search_output, beam_search_state, next_inputs, finished)


def _beam_search_step(time, logits, next_cell_state, beam_state, batch_size,
                      beam_width, end_token, length_penalty_weight,
                      coverage_penalty_weight, max_tgt):
  """Performs a single step of Beam Search Decoding.

  Args:
    time: Beam search time step, should start at 0. At time 0 we assume
      that all beams are equal and consider only the first beam for
      continuations.
    logits: Logits at the current time step. A tensor of shape
      `[batch_size, beam_width, vocab_size]`
    next_cell_state: The next state from the cell, e.g. an instance of
      AttentionWrapperState if the cell is attentional.
    beam_state: Current state of the beam search.
      An instance of `BeamSearchDecoderState`.
    batch_size: The batch size for this input.
    beam_width: Python int.  The size of the beams.
    end_token: The int32 end token.
    length_penalty_weight: Float weight to penalize length. Disabled with 0.0.
    coverage_penalty_weight: Float weight to penalize the coverage of source
      sentence. Disabled with 0.0.
    max_tgt: maximum prediction length.

  Returns:
    A new beam state.
  """

  # Calculate the current lengths of the predictions
  prediction_lengths = beam_state.lengths
  previously_finished = beam_state.finished
  not_finished = tf.logical_not(previously_finished)

  # Calculate the total log probs for the new hypotheses
  # Final Shape: [batch_size, beam_width, vocab_size]
  logits = tf.reshape(logits, [-1, logits.shape[-1]])
  step_log_probs = logits - tf.log(
      tf.reduce_sum(tf.exp(logits), -1, keepdims=True))
  step_log_probs = _mask_probs(step_log_probs, end_token, previously_finished)
  total_probs = tf.reshape(beam_state.log_probs, [-1, 1]) + step_log_probs

  # Calculate the accumulated attention probabilities if coverage penalty is
  # enabled.
  accumulated_attention_probs = None
  attention_probs = get_attention_probs(
      next_cell_state, coverage_penalty_weight)
  if attention_probs is not None:
    attention_probs = tf.reshape(attention_probs, [batch_size * beam_width, -1])
    attention_probs *= tf.expand_dims(
        tf.cast(not_finished, attention_probs.dtype), 1)
    accumulated_attention_probs = (
        tf.reshape(beam_state.accumulated_attention_probs,
                   [batch_size * beam_width, -1]) + attention_probs)

  batch_finished = tf.reduce_all(
      tf.reshape(beam_state.finished, [batch_size, beam_width]),
      axis=1,
      keepdims=True)

  def _normalized_scores(total_probs):
    """Compute normalized scores."""
    # Calculate the continuation lengths by adding to all continuing beams.
    vocab_size = logits.shape[-1].value or tf.shape(logits)[-1]
    lengths_to_add = tf.one_hot(
        indices=tf.fill([batch_size * beam_width], end_token),
        depth=vocab_size,
        on_value=np.int32(0),
        off_value=np.int32(1),
        dtype=tf.int32)
    add_mask = tf.to_int32(not_finished)
    lengths_to_add *= tf.expand_dims(add_mask, 1)
    new_prediction_lengths = (
        lengths_to_add + tf.expand_dims(prediction_lengths, 1))

    return _get_scores(
        log_probs=total_probs,
        sequence_lengths=new_prediction_lengths,
        length_penalty_weight=length_penalty_weight,
        coverage_penalty_weight=coverage_penalty_weight,
        finished=previously_finished,
        accumulated_attention_probs=accumulated_attention_probs)

  def _get_top_k(normalize):
    """Use topk to compute next_word_ids, batch_ids and next_beam_probs."""
    # Pick the next beams according to the specified successors function
    if normalize:
      scores = tf.cast(_normalized_scores(total_probs), tf.float32)
    else:
      scores = tf.cast(total_probs, tf.float32)
    next_beam_scores, word_indices = tf.math.top_k(scores, beam_width)
    scores_flat = tf.reshape(next_beam_scores, [batch_size, -1])
    next_beam_scores, new_indices = tf.math.top_k(scores_flat, beam_width)
    next_beam_scores = tf.cast(next_beam_scores, total_probs.dtype)
    next_beam_ids = tf.div(new_indices, beam_width)
    next_beam_ids = tf.where(
        tf.tile(batch_finished, [1, beam_width]),
        tf.tile(
            tf.reshape(tf.range(beam_width, dtype=tf.int32), [1, -1]),
            [batch_size, 1]), next_beam_ids)
    next_word_ids = tf.gather(
        tf.reshape(word_indices, [batch_size, -1]), new_indices, batch_dims=1)
    next_word_ids = tf.reshape(next_word_ids, [-1])
    batch_ids = tf.reshape(
        tf.expand_dims(tf.range(batch_size) * beam_width, 1) + next_beam_ids,
        [-1])
    if normalize:
      indices = tf.concat(
          [tf.expand_dims(batch_ids, 1),
           tf.expand_dims(next_word_ids, 1)], -1)
      next_beam_probs = tf.gather_nd(total_probs, indices)
    else:
      next_beam_probs = tf.reshape(next_beam_scores, [-1])
    return next_word_ids, batch_ids, next_beam_probs

  next_word_ids, batch_ids, next_beam_probs = _get_top_k(False)

  previously_finished = tf.gather(previously_finished, batch_ids)
  next_finished = tf.logical_or(
      previously_finished,
      tf.equal(next_word_ids, end_token),
      name="next_beam_finished")

  # Calculate the length of the next predictions.
  # 1. Finished beams remain unchanged.
  # 2. Beams that are now finished (EOS predicted) have their length
  #    increased by 1.
  # 3. Beams that are not yet finished have their length increased by 1.
  lengths_to_add = tf.to_int32(tf.logical_not(previously_finished))
  next_prediction_len = tf.gather(beam_state.lengths, batch_ids)
  next_prediction_len += lengths_to_add
  next_accumulated_attention_probs = ()
  if accumulated_attention_probs is not None:
    next_accumulated_attention_probs = tf.gather(accumulated_attention_probs,
                                                 batch_ids)
  next_pred_ids = tf.gather(beam_state.pred_ids, batch_ids)

  # Add next_word_ids to next_pred_ids.
  time = tf.convert_to_tensor(time, name="time")
  time_mask = tf.tile(
      tf.reshape(tf.equal(tf.range(max_tgt), time), [1, max_tgt]),
      [batch_size * beam_width, 1])
  cur_time_ids = tf.tile(
      tf.reshape(next_word_ids, [batch_size * beam_width, 1]), [1, max_tgt])
  next_pred_ids = tf.where(time_mask, cur_time_ids, next_pred_ids)

  # Pick out the cell_states according to the next_beam_ids. We use a
  # different gather_shape here because the cell_state tensors, i.e.
  # the tensors that would be gathered from, all have dimension
  # greater than two and we need to preserve those dimensions.
  # pylint: disable=g-long-lambda
  next_cell_state = tf.contrib.framework.nest.map_structure(
      lambda gather_from: tf.gather(gather_from, batch_ids), next_cell_state)

  next_state = BeamSearchDecoderState(
      cell_state=next_cell_state,
      log_probs=next_beam_probs,
      lengths=next_prediction_len,
      finished=next_finished,
      accumulated_attention_probs=next_accumulated_attention_probs,
      pred_ids=next_pred_ids)

  return next_word_ids, next_state


def get_attention_probs(next_cell_state, coverage_penalty_weight):
  """Get attention probabilities from the cell state.

  Args:
    next_cell_state: The next state from the cell, e.g. an instance of
      AttentionWrapperState if the cell is attentional.
    coverage_penalty_weight: Float weight to penalize the coverage of source
      sentence. Disabled with 0.0.

  Returns:
    The attention probabilities with shape `[batch_size, beam_width, max_time]`
    if coverage penalty is enabled. Otherwise, returns None.

  Raises:
    ValueError: If no cell is attentional but coverage penalty is enabled.
  """
  if coverage_penalty_weight == 0.0:
    return None

  # Attention probabilities of each attention layer. Each with shape
  # `[batch_size, beam_width, max_time]`.
  probs_per_attn_layer = [attention_probs_from_attn_state(next_cell_state[0])]
  for state in next_cell_state:
    if "attention" in state:
      probs_per_attn_layer.append(attention_probs_from_attn_state(state))

  if not probs_per_attn_layer:
    raise ValueError(
        "coverage_penalty_weight must be 0.0 if no cell is attentional.")

  if len(probs_per_attn_layer) == 1:
    attention_probs = probs_per_attn_layer[0]
  else:
    # Calculate the average attention probabilities from all attention layers.
    attention_probs = [
        tf.expand_dims(prob, -1) for prob in probs_per_attn_layer]
    attention_probs = tf.concat(attention_probs, -1)
    attention_probs = tf.reduce_mean(attention_probs, -1)

  return attention_probs


def _get_scores(log_probs, sequence_lengths, length_penalty_weight,
                coverage_penalty_weight, finished, accumulated_attention_probs):
  """Calculates scores for beam search hypotheses.

  Args:
    log_probs: The log probabilities with shape
      `[batch_size, beam_width, vocab_size]`.
    sequence_lengths: The array of sequence lengths.
    length_penalty_weight: Float weight to penalize length. Disabled with 0.0.
    coverage_penalty_weight: Float weight to penalize the coverage of source
      sentence. Disabled with 0.0.
    finished: A boolean tensor of shape `[batch_size, beam_width, vocab_size]`
      that specifies which elements in the beam are finished already.
    accumulated_attention_probs: Accumulated attention probabilities up to the
      current time step, with shape `[batch_size, beam_width, max_time]` if
      coverage_penalty_weight is not 0.0.

  Returns:
    The scores normalized by the length_penalty and coverage_penalty.

  Raises:
    ValueError: accumulated_attention_probs is None when coverage penalty is
      enabled.
  """
  length_penalty_ = _length_penalty(
      sequence_lengths=sequence_lengths, penalty_factor=length_penalty_weight,
      dtype=log_probs.dtype)

  if coverage_penalty_weight == 0.0:
    return log_probs / length_penalty_

  coverage_penalty_weight = tf.convert_to_tensor(
      coverage_penalty_weight, name="coverage_penalty_weight",
      dtype=log_probs.dtype)
  if coverage_penalty_weight.shape.ndims != 0:
    raise ValueError("coverage_penalty_weight should be a scalar, "
                     "but saw shape: %s" % coverage_penalty_weight.shape)

  if accumulated_attention_probs is None:
    raise ValueError(
        "accumulated_attention_probs can be None only if coverage penalty is "
        "disabled.")

  # Add source sequence length mask before computing coverage penalty.
  accumulated_attention_probs = tf.where(
      tf.equal(accumulated_attention_probs, 0.0),
      tf.ones_like(accumulated_attention_probs),
      accumulated_attention_probs)

  # coverage penalty =
  #     sum over `max_time` {log(min(accumulated_attention_probs, 1.0))}
  coverage_penalty = tf.reduce_sum(
      tf.log(tf.minimum(accumulated_attention_probs, 1.0)), 1)
  # Apply coverage penalty to finished predictions.
  weighted_coverage_penalty = coverage_penalty * coverage_penalty_weight
  # Reshape from [batch_size, beam_width] to [batch_size, beam_width, 1]
  weighted_coverage_penalty = tf.expand_dims(weighted_coverage_penalty, 1)

  # Normalize the scores of finished predictions.
  return tf.where(
      finished, log_probs / length_penalty_ + weighted_coverage_penalty,
      log_probs)


def attention_probs_from_attn_state(attention_state):
  """Calculates the average attention probabilities.

  Args:
    attention_state: An instance of `AttentionWrapperState`.

  Returns:
    The attention probabilities in the given AttentionWrapperState.
    If there're multiple attention mechanisms, return the average value from
    all attention mechanisms.
  """
  # Attention probabilities over time steps, with shape
  # `[batch_size, beam_width, max_time]`.
  attention_probs = attention_state.alignments
  if isinstance(attention_probs, tuple):
    attention_probs = [
        tf.expand_dims(prob, -1) for prob in attention_probs]
    attention_probs = tf.concat(attention_probs, -1)
    attention_probs = tf.reduce_mean(attention_probs, -1)
  return attention_probs


def _length_penalty(sequence_lengths, penalty_factor, dtype):
  """Calculates the length penalty. See https://arxiv.org/abs/1609.08144.

  Returns the length penalty tensor:
  ```
  [(5+sequence_lengths)/6]**penalty_factor
  ```
  where all operations are performed element-wise.

  Args:
    sequence_lengths: `Tensor`, the sequence lengths of each hypotheses.
    penalty_factor: A scalar that weights the length penalty.
    dtype: dtype of result.

  Returns:
    If the penalty is `0`, returns the scalar `1.0`.  Otherwise returns
    the length penalty factor, a tensor with the same shape as
    `sequence_lengths`.
  """
  penalty_factor = tf.convert_to_tensor(
      penalty_factor, name="penalty_factor", dtype=dtype)
  penalty_factor.set_shape(())  # penalty should be a scalar.
  static_penalty = tf.contrib.util.constant_value(penalty_factor)
  if static_penalty is not None and static_penalty == 0:
    return 1.0
  length_penalty_const = 5.0
  return tf.div((length_penalty_const + tf.cast(sequence_lengths, dtype))
                **penalty_factor, (length_penalty_const + 1.)**penalty_factor)


def _mask_probs(probs, eos_token, finished):
  """Masks log probabilities.

  The result is that finished beams allocate all probability mass to eos and
  unfinished beams remain unchanged.

  Args:
    probs: Log probabilities of shape `[batch_size, beam_width, vocab_size]`
    eos_token: An int32 id corresponding to the EOS token to allocate
      probability to.
    finished: A boolean tensor of shape `[batch_size, beam_width]` that
      specifies which elements in the beam are finished already.

  Returns:
    A tensor of shape `[batch_size, beam_width, vocab_size]`, where unfinished
    beams stay unchanged and finished beams are replaced with a tensor with all
    probability on the EOS token.
  """
  vocab_size = tf.shape(probs)[1]
  # All finished examples are replaced with a vector that has all
  # probability on EOS
  finished_row = tf.one_hot(
      eos_token,
      vocab_size,
      dtype=probs.dtype,
      on_value=tf.convert_to_tensor(0., dtype=probs.dtype),
      off_value=probs.dtype.min)
  finished_probs = tf.tile(
      tf.reshape(finished_row, [1, -1]), tf.concat([tf.shape(finished), [1]],
                                                   0))
  finished_mask = tf.tile(tf.expand_dims(finished, 1), [1, vocab_size])

  return tf.where(finished_mask, finished_probs, probs)
