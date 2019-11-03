#! /usr/bin/env python3
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

import ctypes
import os
import sys

# The plugin .so file has to be loaded at global scope and before `import torch` to avoid cuda version mismatch.
NMS_OPT_PLUGIN_LIBRARY="build/plugins/NMSOptPlugin/libnmsoptplugin.so"
if not os.path.isfile(NMS_OPT_PLUGIN_LIBRARY):
    raise IOError("{}\n{}\n".format(
        "Failed to load library ({}).".format(NMS_OPT_PLUGIN_LIBRARY),
        "Please build the NMS Opt plugin."
    ))
ctypes.CDLL(NMS_OPT_PLUGIN_LIBRARY)

sys.path.append(os.getcwd())

import argparse
import enum
import json
import numpy as np
import pytest
import tensorrt as trt
import time

from code.common.runner import EngineRunner, get_input_format
from code.common import logging
import code.common.arguments as common_args
from glob import glob
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# The output detections for each image is [keepTopK, 7]. The 7 elements are:
class PredictionLayout(enum.IntEnum):
    IMAGE_ID = 0
    YMIN = 1
    XMIN = 2
    YMAX = 3
    XMAX = 4
    CONFIDENCE = 5
    LABEL = 6

def run_SSDMobileNet_accuracy(engine_file, batch_size, num_images, verbose=False, output_file="build/out/SSDMobileNet/dump.json"):
    logging.info("Running SSDMobileNet functionality test for engine [ {:} ] with batch size {:}".format(engine_file, batch_size))

    runner = EngineRunner(engine_file, verbose=verbose)
    input_dtype, input_format = get_input_format(runner.engine)
    if input_dtype == trt.DataType.FLOAT:
        format_string = "fp32"
    elif input_dtype == trt.DataType.INT8:
        if input_format == trt.TensorFormat.LINEAR:
            format_string = "int8_linear"
        elif input_format == trt.TensorFormat.CHW4:
            format_string = "int8_chw4"
    image_dir = os.path.join(os.getenv("PREPROCESSED_DATA_DIR", "build/preprocessed_data"),
            "coco/val2017/SSDMobileNet", format_string)
    annotations_path = os.path.join(os.getenv("PREPROCESSED_DATA_DIR", "build/preprocessed_data"),
            "coco/annotations/instances_val2017.json")
    val_map = "data_maps/coco/val_map.txt"

    if len(glob(image_dir)) == 0:
        logging.warn("Cannot find data directory in ({:})".format(image_dir))
        pytest.skip("Cannot find data directory ({:})".format(image_dir))

    coco = COCO(annotation_file=annotations_path)

    coco_detections = []
    image_ids = coco.getImgIds()
    num_images = min(num_images, len(image_ids))

    logging.info("Running validation on {:} images. Please wait...".format(num_images))
    batch_idx = 0
    for image_idx in range(0, num_images, batch_size):
        batch_image_ids = image_ids[image_idx:image_idx + batch_size]
        actual_batch_size = len(batch_image_ids)
        batch_images = np.ascontiguousarray(np.stack([np.load(os.path.join(image_dir, coco.imgs[id]["file_name"] + ".npy")) for id in batch_image_ids]))

        start_time = time.time()
        [outputs] = runner([batch_images], actual_batch_size)
        if verbose:
            logging.info("Batch {:d} >> Inference time:  {:f}".format(batch_idx, time.time() - start_time))

        batch_detections = outputs.reshape(batch_size, 100*7+1)[:actual_batch_size]

        for detections, image_id in zip(batch_detections, batch_image_ids):
            keep_count = detections[100*7].view('int32')
            image_width = coco.imgs[image_id]["width"]
            image_height = coco.imgs[image_id]["height"]
            for detection in detections[:keep_count*7].reshape(keep_count,7):
                score = float(detection[PredictionLayout.CONFIDENCE])
                bbox_coco_fmt = [
                    detection[PredictionLayout.XMIN] * image_width,
                    detection[PredictionLayout.YMIN] * image_height,
                    (detection[PredictionLayout.XMAX] - detection[PredictionLayout.XMIN]) * image_width,
                    (detection[PredictionLayout.YMAX] - detection[PredictionLayout.YMIN]) * image_height,
                ]

                coco_detection = {
                    "image_id": image_id,
                    "category_id": int(detection[PredictionLayout.LABEL]),
                    "bbox": bbox_coco_fmt,
                    "score": score,
                }
                coco_detections.append(coco_detection)

        batch_idx += 1

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, "w") as f:
        json.dump(coco_detections, f)

    cocoDt = coco.loadRes(output_file)
    eval = COCOeval(coco, cocoDt, 'bbox')
    eval.params.imgIds = image_ids[:num_images]

    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    map_score = eval.stats[0]

    logging.info("Get mAP score = {:f} Target = {:f}".format(map_score, 0.22386))
    return map_score

def main():
    args = common_args.parse_args(common_args.ACCURACY_ARGS)
    logging.info("Running accuracy test...")
    run_SSDMobileNet_accuracy(args["engine_file"], args["batch_size"], args["num_images"],
            verbose=args["verbose"])

if __name__ == "__main__":
    main()
