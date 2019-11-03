# Copyright 2018 Google. All Rights Reserved.
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
"""Model defination for the SSD Model.

Defines model_fn of SSD for TF Estimator. The model_fn includes SSD
model architecture, loss function, learning rate schedule, and evaluation
procedure.

T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar
Focal Loss for Dense Object Detection. arXiv:1708.02002
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools as it
import math
import numpy as np
import tensorflow as tf

import ssd_architecture
import ssd_constants

from tensorflow.contrib.tpu.python.tpu import bfloat16


BBOX_XFORM_CLIP = np.log(1000. / 16.)


class DefaultBoxes(object):
  """Default bounding boxes for 1200x1200 5 layer SSD.

  Default bounding boxes generation follows the order of (W, H, anchor_sizes).
  Therefore, the tensor converted from DefaultBoxes has a shape of
  [anchor_sizes, H, W, 4]. The last dimension is the box coordinates; 'ltrb'
  is [ymin, xmin, ymax, xmax] while 'xywh' is [cy, cx, h, w].
  """

  def __init__(self):
    steps = [
        int(ssd_constants.IMAGE_SIZE / fs) for fs in ssd_constants.FEATURE_SIZES
    ]
    fk = ssd_constants.IMAGE_SIZE / np.array(steps)

    self.default_boxes = []
    # Scale by image size.
    scales = [
        int(s * ssd_constants.IMAGE_SIZE / 300) for s in ssd_constants.SCALES
    ]
    # size of feature and number of feature
    for idx, feature_size in enumerate(ssd_constants.FEATURE_SIZES):
      sk1 = scales[idx] / ssd_constants.IMAGE_SIZE
      sk2 = scales[idx + 1] / ssd_constants.IMAGE_SIZE
      sk3 = math.sqrt(sk1 * sk2)
      all_sizes = [(sk1, sk1), (sk3, sk3)]

      for alpha in ssd_constants.ASPECT_RATIOS[idx]:
        w, h = sk1 * math.sqrt(alpha), sk1 / math.sqrt(alpha)
        all_sizes.append((w, h))
        all_sizes.append((h, w))

      assert len(all_sizes) == ssd_constants.NUM_DEFAULTS[idx]

      for w, h in all_sizes:
        for i, j in it.product(range(feature_size), repeat=2):
          cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
          box = tuple(np.clip(k, 0, 1) for k in (cy, cx, h, w))
          self.default_boxes.append(box)

    assert len(self.default_boxes) == ssd_constants.NUM_SSD_BOXES

    def to_ltrb(cy, cx, h, w):
      return cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2

    # For IoU calculation
    self.default_boxes_ltrb = tuple(to_ltrb(*i) for i in self.default_boxes)

  def __call__(self, order='ltrb'):
    if order == 'ltrb':
      return self.default_boxes_ltrb
    if order == 'xywh':
      return self.default_boxes


def decode_boxes(encoded_boxes, anchors, weights=None):
  """Decode boxes.

  Args:
    encoded_boxes: a tensor whose last dimension is 4 representing the
      coordinates of encoded boxes in ymin, xmin, ymax, xmax order.
    anchors: a tensor whose shape is the same as `boxes` representing the
      coordinates of anchors in ymin, xmin, ymax, xmax order.
    weights: None or a list of four float numbers used to scale coordinates.

  Returns:
    encoded_boxes: a tensor whose shape is the same as `boxes` representing the
      decoded box targets.
  """
  with tf.name_scope('decode_box'):
    encoded_boxes = tf.cast(encoded_boxes, dtype=anchors.dtype)
    dy = encoded_boxes[..., 0:1]
    dx = encoded_boxes[..., 1:2]
    dh = encoded_boxes[..., 2:3]
    dw = encoded_boxes[..., 3:4]
    if weights:
      dy /= weights[0]
      dx /= weights[1]
      dh /= weights[2]
      dw /= weights[3]
    dh = tf.minimum(dh, BBOX_XFORM_CLIP)
    dw = tf.minimum(dw, BBOX_XFORM_CLIP)

    anchor_ymin = anchors[..., 0:1]
    anchor_xmin = anchors[..., 1:2]
    anchor_ymax = anchors[..., 2:3]
    anchor_xmax = anchors[..., 3:4]

    anchor_h = anchor_ymax - anchor_ymin
    anchor_w = anchor_xmax - anchor_xmin
    anchor_yc = anchor_ymin + 0.5 * anchor_h
    anchor_xc = anchor_xmin + 0.5 * anchor_w

    decoded_boxes_yc = dy * anchor_h + anchor_yc
    decoded_boxes_xc = dx * anchor_w + anchor_xc
    decoded_boxes_h = tf.exp(dh) * anchor_h
    decoded_boxes_w = tf.exp(dw) * anchor_w

    decoded_boxes_ymin = decoded_boxes_yc - 0.5 * decoded_boxes_h
    decoded_boxes_xmin = decoded_boxes_xc - 0.5 * decoded_boxes_w
    decoded_boxes_ymax = decoded_boxes_yc + 0.5 * decoded_boxes_h
    decoded_boxes_xmax = decoded_boxes_xc + 0.5 * decoded_boxes_w

    decoded_boxes = tf.concat([
        decoded_boxes_ymin, decoded_boxes_xmin, decoded_boxes_ymax,
        decoded_boxes_xmax
    ],
                              axis=-1)
    return decoded_boxes


def select_top_k_scores(scores_in, pre_nms_num_detections=5000):
  """Select top_k scores and indices for each class.

  Args:
    scores_in: a Tensor with shape [batch_size, num_classes, N], which stacks
      class logit outputs on all feature levels. The N is the number of total
      anchors on all levels. The num_classes is the number of classes predicted
      by the model.
    pre_nms_num_detections: Number of candidates before NMS.

  Returns:
    scores and indices: Tensors with shape [batch_size, pre_nms_num_detections,
      num_classes].
  """
  _, num_class, num_anchors = scores_in.get_shape().as_list()

  scores = tf.reshape(scores_in, [-1, num_anchors])

  top_k_scores, top_k_indices = tf.nn.top_k(
      scores, k=pre_nms_num_detections, sorted=True)

  top_k_scores = tf.reshape(top_k_scores,
                            [-1, num_class, pre_nms_num_detections])
  top_k_indices = tf.reshape(top_k_indices,
                             [-1, num_class, pre_nms_num_detections])

  return tf.transpose(top_k_scores, [0, 2, 1]), tf.transpose(
      top_k_indices, [0, 2, 1])


def _filter_scores(scores, boxes, min_score=ssd_constants.MIN_SCORE):
  mask = scores > min_score
  scores = tf.where(mask, scores, tf.zeros_like(scores))
  boxes = tf.where(
      tf.tile(tf.expand_dims(mask, 2), (1, 1, 4)), boxes, tf.zeros_like(boxes))
  return scores, boxes


def non_max_suppression(scores_in,
                        boxes_in,
                        top_k_indices,
                        source_id,
                        raw_shape,
                        num_detections=ssd_constants.MAX_NUM_EVAL_BOXES):
  """Implement Non-maximum suppression.

  Args:
    scores_in: a Tensor with shape [batch_size,
      ssd_constants.MAX_NUM_EVAL_BOXES, num_classes]. The top
      ssd_constants.MAX_NUM_EVAL_BOXES box scores for each class.
    boxes_in: a Tensor with shape [batch_size, N, 4], which stacks box
      regression outputs on all feature levels. The N is the number of total
      anchors on all levels.
    top_k_indices: a Tensor with shape [batch_size,
      ssd_constants.MAX_NUM_EVAL_BOXES, num_classes]. The indices for these top
      boxes for each class.
    source_id: a Tensor with shape [batch_size]
    raw_shape: a Tensor with shape [batch_size, 3]
    num_detections: maximum output length.

  Returns:
    A tensor size of [batch_size, num_detections, 6] represents boxes, labels
    and scores after NMS.
  """

  _, _, num_classes = scores_in.get_shape().as_list()
  source_id = tf.to_float(
      tf.tile(tf.expand_dims(source_id, 1), [1, num_detections]))
  raw_shape = tf.to_float(
      tf.tile(tf.expand_dims(raw_shape, 1), [1, num_detections, 1]))

  list_of_all_boxes = []
  list_of_all_scores = []
  list_of_all_classes = []
  # Skip background class.
  for class_i in range(1, num_classes, 1):
    boxes = tf.batch_gather(boxes_in, top_k_indices[:, :, class_i])
    class_i_scores = scores_in[:, :, class_i]
    class_i_scores, boxes = _filter_scores(class_i_scores, boxes)
    (class_i_post_scores,
     class_i_post_boxes) = ssd_architecture.non_max_suppression_padded(
         scores=tf.to_float(class_i_scores),
         boxes=tf.to_float(boxes),
         max_output_size=num_detections,
         iou_threshold=ssd_constants.OVERLAP_CRITERIA)
    class_i_classes = tf.fill(tf.shape(class_i_post_scores), class_i)
    list_of_all_boxes.append(class_i_post_boxes)
    list_of_all_scores.append(class_i_post_scores)
    list_of_all_classes.append(class_i_classes)

  post_nms_boxes = tf.concat(list_of_all_boxes, axis=1)
  post_nms_scores = tf.concat(list_of_all_scores, axis=1)
  post_nms_classes = tf.concat(list_of_all_classes, axis=1)

  # sort all results.
  post_nms_scores, sorted_indices = tf.nn.top_k(
      tf.to_float(post_nms_scores), k=num_detections, sorted=True)

  post_nms_boxes = tf.batch_gather(post_nms_boxes, sorted_indices)
  post_nms_classes = tf.batch_gather(post_nms_classes, sorted_indices)
  detections_result = tf.stack([
      source_id,
      post_nms_boxes[:, :, 0],
      post_nms_boxes[:, :, 1],
      post_nms_boxes[:, :, 2],
      post_nms_boxes[:, :, 3],
      post_nms_scores,
      tf.to_float(post_nms_classes),
  ],
                               axis=2)

  return detections_result


def concat_outputs(cls_outputs, box_outputs):
  """Concatenate predictions into a single tensor.

  This function takes the dicts of class and box prediction tensors and
  concatenates them into a single tensor for comparison with the ground truth
  boxes and class labels.
  Args:
    cls_outputs: an OrderDict with keys representing levels and values
      representing logits in [batch_size, height, width,
      num_anchors * num_classses].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in
      [batch_size, height, width, num_anchors * 4].
  Returns:
    concatenanted cls_outputs with shape [batch_size, num_classes, N] and
    concatenanted box_outputs with shape [batch_size, 4, N], where N is number
    of anchors.
  """
  assert set(cls_outputs.keys()) == set(box_outputs.keys())

  # This sort matters. The labels assume a certain order based on
  # ssd_constants.FEATURE_SIZES, and this sort matches that convention.
  keys = sorted(cls_outputs.keys())

  flat_cls = []
  flat_box = []

  for i, k in enumerate(keys):
    # TODO(taylorrobie): confirm that this reshape, transpose,
    # reshape is correct.
    scale = ssd_constants.FEATURE_SIZES[i]
    last_dim_size = scale * scale * ssd_constants.NUM_DEFAULTS[i]
    split_shape = (ssd_constants.NUM_CLASSES, ssd_constants.NUM_DEFAULTS[i])
    assert cls_outputs[k].shape[3] == split_shape[0] * split_shape[1]
    flat_cls.append(
        tf.reshape(
            tf.transpose(cls_outputs[k], [0, 3, 1, 2]),
            [-1, ssd_constants.NUM_CLASSES, last_dim_size]))

    split_shape = (ssd_constants.NUM_DEFAULTS[i], 4)
    assert box_outputs[k].shape[3] == split_shape[0] * split_shape[1]
    flat_box.append(
        tf.reshape(
            tf.transpose(box_outputs[k], [0, 3, 1, 2]), [-1, 4, last_dim_size]))

  return tf.concat(flat_cls, axis=2), tf.concat(flat_box, axis=2)


def _model_fn(images, source_id, raw_shape, params, model):
  """Model defination for the SSD model based on ResNet-50.

  Args:
    images: the input image tensor with shape [batch_size, height, width, 3].
      The height and width are fixed and equal.
    source_id: a Tensor with shape [batch_size]
    raw_shape: a Tensor with shape [batch_size, 3]
    params: the dictionary defines hyperparameters of model. The default
      settings are in default_hparams function in this file.
    model: the SSD model outputs class logits and box regression outputs.

  Returns:
    spec: the EstimatorSpec or TPUEstimatorSpec to run training, evaluation,
      or prediction.
  """
  features = images

  def _model_outputs():
    return model(features, params, is_training_bn=False)

  if params['use_bfloat16']:
    with bfloat16.bfloat16_scope():
      cls_outputs, box_outputs = _model_outputs()
      levels = cls_outputs.keys()
      for level in levels:
        cls_outputs[level] = tf.cast(cls_outputs[level], tf.float32)
        box_outputs[level] = tf.cast(box_outputs[level], tf.float32)
  else:
    cls_outputs, box_outputs = _model_outputs()
    levels = cls_outputs.keys()

  flattened_cls, flattened_box = concat_outputs(cls_outputs, box_outputs)

  y_min, x_min, y_max, x_max = tf.split(flattened_box, 4, axis=1)
  flattened_box = tf.concat([x_min, y_min, x_max, y_max], axis=1)
  # [batch_size, 4, N] to [batch_size, N, 4]
  flattened_box = tf.transpose(flattened_box, [0, 2, 1])

  anchors = tf.convert_to_tensor(DefaultBoxes()('ltrb'))

  decoded_boxes = decode_boxes(
      encoded_boxes=flattened_box,
      anchors=anchors,
      weights=ssd_constants.BOX_CODER_SCALES)

  pred_scores = tf.nn.softmax(flattened_cls, axis=1)
  pred_scores, indices = select_top_k_scores(pred_scores,
                                             ssd_constants.MAX_NUM_EVAL_BOXES)
  detections = non_max_suppression(
      scores_in=pred_scores,
      boxes_in=decoded_boxes,
      top_k_indices=indices,
      source_id=source_id,
      raw_shape=raw_shape)

  return detections


def ssd_model_fn(images, source_id, raw_shape, params):
  """SSD model."""
  return _model_fn(
      images, source_id, raw_shape, params, model=ssd_architecture.ssd)


def default_hparams():
  return tf.contrib.training.HParams(
      use_bfloat16=True,
      transpose_input=True,
      nms_on_tpu=True,
      conv0_space_to_depth=False,
      use_cocoeval_cc=True,
      use_spatial_partitioning=False,
  )
