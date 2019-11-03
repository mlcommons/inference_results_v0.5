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

import numpy as np
import sys
import os
import io
import argparse
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
import torch
from random import shuffle
import re

def dump_image(tensor, fileName):
    bytes=np.array(tensor, np.dtype('uint8')).tobytes()
    if len(bytes)!=150528:
        raise ValueError("Bad image! Size should be 150528 bytes")
    if not os.path.exists(os.path.dirname(fileName)):
        os.makedirs(os.path.dirname(fileName))
    np.save(fileName, bytes)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

def process(data_dir, output_dir, data_map, overwrite):
    lines=[]
    with open(data_map, "r") as mapFile:
        for line in mapFile:
            lines.append(line.strip())
    shuffle(lines)
    for line in lines:
        split=line.strip().replace("\t", " ").split(" ")
        filePath=split[0]
        print("Processing " + filePath)
        result = re.match(r".*(ILSVRC2012_val_000\d+.JPEG)", filePath)
        fileName = result.group(1)

        in_file = os.path.join(data_dir, "imagenet", fileName)
        out_file = os.path.join(output_dir, "imagenet", "ResNet50_int4", fileName)

        if not overwrite and os.path.exists(out_file):
            print("Skipping {:} because it exists...".format(out_file))

        img = Image.open(in_file)
        if img.mode!="RGB":
            img=img.convert("RGB")
        img = preprocess(img)
        tensor = torch.clamp(torch.round(img*127.0/2.64), -127.0, 127.0)
        dump_image(tensor, out_file)

def main():
    # Parse arguments to identify the data directory with the input images
    #   and the output directory for the preprocessed images.
    # The data dicretory is assumed to have the following structure:
    # <data_directory>
    #  └── imagenet
    # And the output directory will have the following structure:
    # <output_directory>
    #  └── imagenet
    #      └── ResNet50_int4
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
        "--data_map", "-m",
        help="Specifies the output directory for the preprocessed data.",
        default="data_maps/imagenet/val_map.txt"
    )
    parser.add_argument(
        "--overwrite", "-f",
        help="Overwrite existing files.",
        action="store_true"
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    data_map = args.data_map
    overwrite = args.overwrite
    process(data_dir, output_dir, data_map, overwrite)

    print("Preprocessing done!")

if __name__ == "__main__":
    main()
