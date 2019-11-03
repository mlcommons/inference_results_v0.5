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

import os
import sys
sys.path.insert(0, os.getcwd())

import argparse
import numpy as np
import shutil

from code.common import logging
import cv2
import math

def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    
    img = img[top:bottom, left:right]
    return img

def resize_with_aspectratio(img, out_height, out_width, scale=87.5):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img

class ImagePreprocessor:
    def __init__(self, loader, quantizer):
        self.loader = loader
        self.quantizer = quantizer
        self.all_formats = ["fp32", "int8_linear", "int8_chw4"]

    def run(self, src_dir, dst_dir, data_map, formats, overwrite=False):
        assert all([i in self.all_formats for i in formats]), "Unsupported formats {:}.".format(formats)
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.run_formats = formats
        self.overwrite = overwrite
        self.make_dirs()

        image_list = []
        with open(data_map) as f:
            for line in f:
                image_list.append(line.split()[0])
        self.convert(image_list)

    def make_dirs(self):
        for format in self.run_formats:
            dir = self.get_dir(format)
            if not os.path.exists(dir):
                os.makedirs(dir)

    def convert(self, image_list):
        for idx, img_file in enumerate(image_list):
            logging.info("Processing image No.{:d}/{:d}...".format(idx, len(image_list)))
            output_files = [self.get_filename(format, img_file) for format in self.run_formats]

            if all([os.path.exists(i) for i in output_files]) and not self.overwrite:
                logging.info("Skipping {:} because it already exists.".format(img_file))
                continue

            image_fp32 = self.loader(os.path.join(self.src_dir, img_file))
            if "fp32" in self.run_formats:
                np.save(self.get_filename("fp32", img_file), image_fp32)
            image_int8_linear = self.quantizer(image_fp32)
            if "int8_linear" in self.run_formats:
                np.save(self.get_filename("int8_linear", img_file), image_int8_linear)
            image_int8_chw4 = self.linear_to_chw4(image_int8_linear)
            if "int8_chw4" in self.run_formats:
                np.save(self.get_filename("int8_chw4", img_file), image_int8_chw4)

    def get_dir(self, format):
        return os.path.join(self.dst_dir, format)

    def get_filename(self, format, img_file):
        return os.path.join(self.get_dir(format), "{:}.npy".format(img_file))

    def linear_to_chw4(self, image_int8):
        return np.moveaxis(np.pad(image_int8, ((0, 1), (0, 0),(0, 0)), "constant"), -3, -1)

def preprocess_imagenet_for_resnet50(data_dir, preprocessed_data_dir, formats, overwrite=False, cal_only=False, val_only=False):
    def loader(file):
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        w, h = (224, 224)
        image = resize_with_aspectratio(image, h, w)
        image = center_crop(image, h, w)
        image = np.asarray(image, dtype='float32')
        # Normalize image.
        means = np.array([123.68, 116.78, 103.94], dtype=np.float32)
        image -= means
        # Transpose.
        image = image.transpose([2, 0, 1])
        return image
    def quantizer(image):
        return np.clip(image, -128.0, 127.0).astype(dtype=np.int8, order='C')
    preprocessor = ImagePreprocessor(loader, quantizer)
    if not val_only:
        # Preprocess calibration set. FP32 only because calibrator always takes FP32 input.
        preprocessor.run(os.path.join(data_dir, "imagenet"), os.path.join(preprocessed_data_dir, "imagenet", "ResNet50"),
            "data_maps/imagenet/cal_map.txt", ["fp32"], overwrite)
    if not cal_only:
        # Preprocess validation set.
        preprocessor.run(os.path.join(data_dir, "imagenet"), os.path.join(preprocessed_data_dir, "imagenet", "ResNet50"),
            "data_maps/imagenet/val_map.txt", formats, overwrite)

def preprocess_imagenet_for_mobilenet(data_dir, preprocessed_data_dir, formats, overwrite=False, cal_only=False, val_only=False):
    def loader(file):
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        w, h = (224, 224)
        image = resize_with_aspectratio(image, h, w)
        image = center_crop(image, h, w)
        image = np.asarray(image, dtype='float32')
        # Normalize image
        means = np.array([128,128,128], dtype=np.float32)
        image -= means
        # Transpose
        image = image.transpose([2,0,1])
        return image
    def quantizer(image):
        # Dynamic range of image is already [-127.0, 127.0]
        return image.astype(dtype=np.int8, order='C')
    preprocessor = ImagePreprocessor(loader, quantizer)
    if not val_only:
        # Preprocess calibration set. FP32 only because calibrator always takes FP32 input.
        preprocessor.run(os.path.join(data_dir, "imagenet"), os.path.join(preprocessed_data_dir, "imagenet", "MobileNet"),
            "data_maps/imagenet/cal_map.txt", ["fp32"], overwrite)
    if not cal_only:
        # Preprocess validation set.
        preprocessor.run(os.path.join(data_dir, "imagenet"), os.path.join(preprocessed_data_dir, "imagenet", "MobileNet"),
            "data_maps/imagenet/val_map.txt", formats, overwrite)

def preprocess_coco_for_ssdmobilenet(data_dir, preprocessed_data_dir, formats, overwrite=False, cal_only=False, val_only=False):
    def loader(file):
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(cv2.resize(image, (300, 300), interpolation=cv2.INTER_LINEAR)).astype(np.float32)
        image = image.transpose((2, 0, 1))
        image = (2.0 / 255.0) * image - 1.0
        return image
    def quantizer(image):
        # Dynamic range of image is [-1.0, 1.0]
        image_int8 = image * 127.0
        return image_int8.astype(dtype=np.int8, order='C')
    preprocessor = ImagePreprocessor(loader, quantizer)
    if not val_only:
        # Preprocess calibration set. FP32 only because calibrator always takes FP32 input.
        preprocessor.run(os.path.join(data_dir, "coco", "train2017"),
            os.path.join(preprocessed_data_dir, "coco", "train2017", "SSDMobileNet"),
            "data_maps/coco/cal_map.txt", ["fp32"], overwrite)
    if not cal_only:
        # Preprocess validation set.
        preprocessor.run(os.path.join(data_dir, "coco", "val2017"),
            os.path.join(preprocessed_data_dir, "coco", "val2017", "SSDMobileNet"),
            "data_maps/coco/val_map.txt", formats, overwrite)

def preprocess_coco_for_ssdresnet34(data_dir, preprocessed_data_dir, formats, overwrite=False, cal_only=False, val_only=False):
    def loader(file):
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(cv2.resize(image, (1200, 1200), interpolation=cv2.INTER_LINEAR)).astype(np.float32)
        image = image.transpose((2, 0, 1))
        image /= 255.0
        # Normalize image.
        means = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, np.newaxis, np.newaxis]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, np.newaxis, np.newaxis]
        image = (image - means) / std
        return image
    def quantizer(image):
        # Dynamic range of image is [-2.64064, 2.64064] based on calibration cache.
        max_abs = 2.64064
        image_int8 = image.clip(-max_abs, max_abs) / max_abs * 127.0
        return image_int8.astype(dtype=np.int8, order='C')
    preprocessor = ImagePreprocessor(loader, quantizer)
    if not val_only:
        # Preprocess calibration set. FP32 only because calibrator always takes FP32 input.
        preprocessor.run(os.path.join(data_dir, "coco", "train2017"),
            os.path.join(preprocessed_data_dir, "coco", "train2017", "SSDResNet34"),
            "data_maps/coco/cal_map.txt", ["fp32"], overwrite)
    if not cal_only:
        # Preprocess validation set.
        preprocessor.run(os.path.join(data_dir, "coco", "val2017"),
            os.path.join(preprocessed_data_dir, "coco", "val2017", "SSDResNet34"),
            "data_maps/coco/val_map.txt", formats, overwrite)

def copy_coco_annotations(data_dir, output_dir):
    src_dir = os.path.join(data_dir, "coco/annotations")
    dst_dir = os.path.join(output_dir, "coco/annotations")
    if not os.path.exists(dst_dir):
        shutil.copytree(src_dir, dst_dir)

def main():
    # Parse arguments to identify the data directory with the input images
    #   and the output directory for the preprocessed images.
    # The data dicretory is assumed to have the following structure:
    # <data_directory>
    #  ├── coco
    #  │   ├── annotations
    #  │   ├── train2017
    #  │   └── val2017
    #  └── imagenet
    # And the output directory will have the following structure:
    # <output_directory>
    #  ├── coco
    #  │   ├── annotations
    #  │   ├── train2017
    #  │   │   ├── SSDMobileNet
    #  │   │   │   └── fp32
    #  │   │   └── SSDResNet34
    #  │   │       └── fp32
    #  │   └── val2017
    #  │       ├── SSDMobileNet
    #  │       │   ├── int8_chw4
    #  │       │   └── int8_linear
    #  │       └── SSDResNet34
    #  │           └── int8_linear
    #  └── imagenet
    #      ├── MobileNet
    #      │   ├── fp32
    #      │   └── int8_chw4
    #      └── ResNet50
    #          ├── fp32
    #          └── int8_linear
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", "-d",
        help="Specifies the directory containing the input images.",
        default=""
    )
    parser.add_argument(
        "--output_dir", "-o",
        help="Specifies the output directory for the preprocessed data.",
        default=""
    )
    parser.add_argument(
        "--benchmarks", "-b",
        help="Comma-separated list of benchmarks. Default: resnet,mobilenet,ssd-large,ssd-small.",
        default="resnet,mobilenet,ssd-large,ssd-small"
    )
    parser.add_argument(
        "--formats", "-t",
        help="Comma-separated list of formats. Choices: fp32, int8_linear, int8_chw4.",
        default="default"
    )
    parser.add_argument(
        "--overwrite", "-f",
        help="Overwrite existing files.",
        action="store_true"
    )
    parser.add_argument(
        "--cal_only",
        help="Only preprocess calibration set.",
        action="store_true"
    )
    parser.add_argument(
        "--val_only",
        help="Only preprocess validation set.",
        action="store_true"
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    benchmarks = args.benchmarks.split(",")
    formats = args.formats.split(",")
    overwrite = args.overwrite
    cal_only = args.cal_only
    val_only = args.val_only
    preprocessor_map = {
        "resnet": preprocess_imagenet_for_resnet50,
        "mobilenet": preprocess_imagenet_for_mobilenet,
        "ssd-small": preprocess_coco_for_ssdmobilenet,
        "ssd-large": preprocess_coco_for_ssdresnet34
    }
    default_formats_map = {
        "resnet": ["int8_linear"],
        "mobilenet": ["int8_chw4"],
        "ssd-small": ["int8_linear", "int8_chw4"],
        "ssd-large": ["int8_linear"]
    }

    # Now, actually preprocess the input images
    logging.info("Loading and preprocessing images. This might take a while...")
    for benchmark in benchmarks:
        if args.formats == "default":
            formats = default_formats_map[benchmark]
        preprocessor_map[benchmark](data_dir, output_dir, formats, overwrite, cal_only, val_only)

    # Copy coco annotations if necessary.
    if "ssd-small" in benchmarks or "ssd-large" in benchmarks:
        copy_coco_annotations(data_dir, output_dir)

    logging.info("Preprocessing done.")

if __name__ == '__main__':
	main()

