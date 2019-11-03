"""implementation of COCO dataset post processing."""

# pylint: disable=unused-argument,missing-docstring

import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class PostProcessCoco(object):
  """Post processing for tensorflow ssd-mobilenet style models."""

  def __init__(self):
    self.results = []
    self.good = 0
    self.total = 0

  def add_results(self, results):
    self.results.extend(results)

  def start(self):
    self.results = []

  def finalize(self, annotation_file=None, output_dir=None):
    image_ids = []

    detections = self.results
    detections = np.reshape(np.array(detections), (-1, 7))
    detections = np.unique(detections, axis=0)

    image_ids = list(set([i[0] for i in detections]))
    self.results = []
    coco_gt = COCO(annotation_file)
    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print("coco_eval stats:", coco_eval.stats)
    return coco_eval.stats[0]
