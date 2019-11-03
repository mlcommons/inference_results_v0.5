#!/usr/bin/env python3
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
import os, sys

# The plugin .so file has to be loaded at global scope and before `import torch` to avoid cuda version mismatch.
NMS_OPT_PLUGIN_LIBRARY="build/plugins/NMSOptPlugin/libnmsoptplugin.so"
if not os.path.isfile(NMS_OPT_PLUGIN_LIBRARY):
    raise IOError("{}\n{}\n".format(
        "Failed to load library ({}).".format(NMS_OPT_PLUGIN_LIBRARY),
        "Please build the NMS Opt plugin."
    ))
ctypes.CDLL(NMS_OPT_PLUGIN_LIBRARY)

import argparse
import json
import time
sys.path.insert(0, os.getcwd())

from code.common.runner import EngineRunner, get_input_format
from code.common import logging
import code.common.arguments as common_args

import numpy as np
import torch
import tensorrt as trt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def run_SSDResNet34_accuracy(engine_file, batch_size, num_images, verbose=False, output_file="build/out/SSDResNet34/dump.json"):
    threshold = 0.20

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
            "coco/val2017/SSDResNet34", format_string)
    val_annotate = os.path.join(os.getenv("PREPROCESSED_DATA_DIR", "build/preprocessed_data"),
            "coco/annotations/instances_val2017.json")

    coco = COCO(annotation_file=val_annotate)

    image_ids = coco.getImgIds()
    cat_ids = coco.getCatIds()
    # Class 0 is background
    cat_ids.insert(0, 0)
    num_images = min(num_images, len(image_ids))

    logging.info("Running validation on {:} images. Please wait...".format(num_images))

    coco_detections = []

    batch_idx = 0
    for image_idx in range(0, num_images, batch_size):
        end_idx = min(image_idx + batch_size, num_images)
        img = []
        img_sizes = []
        for idx in range(image_idx, end_idx):
            image_id = image_ids[idx]
            img.append(np.load(os.path.join(image_dir, coco.imgs[image_id]["file_name"] + ".npy")))
            img_sizes.append([coco.imgs[image_id]["height"], coco.imgs[image_id]["width"]])

        img = np.stack(img)

        start_time = time.time()
        [trt_detections] = runner([img], batch_size=batch_size)
        if verbose:
            logging.info("Batch {:d} >> Inference time:  {:f}".format(batch_idx, time.time() - start_time))

        for idx in range(0, end_idx - image_idx):
            keep_count = trt_detections[idx * (200 * 7 + 1) + 200 * 7].view('int32')
            trt_detections_batch = trt_detections[idx * (200 * 7 + 1):idx * (200 * 7 + 1) + keep_count * 7].reshape(keep_count, 7)
            image_height = img_sizes[idx][0]
            image_width = img_sizes[idx][1]
            for prediction_idx in range(0, keep_count):
                loc = trt_detections_batch[prediction_idx, [2, 1, 4, 3]]
                label = trt_detections_batch[prediction_idx, 6]
                score = float(trt_detections_batch[prediction_idx, 5])

                bbox_coco_fmt = [
                    loc[0] * image_width,
                    loc[1] * image_height,
                    (loc[2] - loc[0]) * image_width,
                    (loc[3] - loc[1]) * image_height,
                ]

                coco_detection = {
                    "image_id": image_ids[image_idx + idx],
                    "category_id": cat_ids[int(label)],
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
    logging.info("Get mAP score = {:f} Target = {:f}".format(map_score, threshold))

    return (map_score >= threshold * 0.99)

def main():
    args = common_args.parse_args(common_args.ACCURACY_ARGS)
    logging.info("Running accuracy test...")
    run_SSDResNet34_accuracy(args["engine_file"], args["batch_size"], args["num_images"],
            verbose=args["verbose"])

if __name__ == "__main__":
    main()
