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

import os, sys
sys.path.append(os.getcwd())

from code.common.runner import EngineRunner, get_input_format
from code.common.accuracy import ImageNetAccuracyRunner
from code.common import logging
import code.common.arguments as common_args
import tensorrt as trt

def run_MobileNet_accuracy(engine_file, batch_size, num_images, verbose=False):
    if verbose:
        logging.info("Running MobileNet accuracy test with:")
        logging.info("    engine_file: {:}".format(engine_file))
        logging.info("    batch_size: {:}".format(batch_size))
        logging.info("    num_images: {:}".format(num_images))

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
            "imagenet/MobileNet", format_string)

    accuracy_runner = ImageNetAccuracyRunner(runner, batch_size, image_dir, num_images,
        verbose=verbose)
    return accuracy_runner.run()

def main():
    args = common_args.parse_args(common_args.ACCURACY_ARGS)
    logging.info("Running accuracy test...")
    acc = run_MobileNet_accuracy(args["engine_file"], args["batch_size"], args["num_images"],
            verbose=args["verbose"])

    logging.info("Accuracy: {:}".format(acc))

if __name__ == "__main__":
    main()
