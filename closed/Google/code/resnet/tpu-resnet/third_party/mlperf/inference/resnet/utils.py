# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""A collection of utilities."""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np

import mlperf_loadgen as lg

from model import backend_tf
from model import dataset
from model import imagenet


BackendTensorflow = backend_tf.BackendTensorflow

# the datasets we support
SUPPORTED_DATASETS = {
    "imagenet": (imagenet.Imagenet, dataset.pre_process_vgg,
                 dataset.PostProcessCommon(offset=0), {
                     "image_size": [4, 224, 224]
                 }),
}

# pre-defined command line options so simplify things. They are used as defaults
# and can be overwritten from command line. (latency in milliseconds)
DEFAULT_LATENCY_BUCKETS = "10,50,100,200,400"

SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")


def setup(dataset_name, dataset_path, dataset_list, cache, cache_dir,
          conv0_space_to_depth_block_size, tpu_transpose,
          data_format=None, count=None):
  """Configures dataset, model, backend, and runner."""

  # define backend
  backend = BackendTensorflow(
      conv0_space_to_depth_block_size=conv0_space_to_depth_block_size,
      tpu_transpose=tpu_transpose)

  # override image format if given
  image_format = data_format if data_format else backend.image_format()

  # dataset to use
  wanted_dataset, pre_proc, post_process, kwargs = SUPPORTED_DATASETS[
      dataset_name]
  ds = wanted_dataset(
      data_path=dataset_path,
      image_list=dataset_list,
      name=dataset_name,
      image_format=image_format,
      need_transpose=True,
      conv0_space_to_depth_block_size=conv0_space_to_depth_block_size,
      pre_process=pre_proc,
      use_cache=cache,
      count=count,
      cache_dir=cache_dir,
      **kwargs)
  #
  # make one pass over the dataset to validate accuracy
  #
  count = count if count else ds.get_item_count()

  return backend, ds, post_process, count


def add_results(final_results, name, result_dict, result_list, took):
  """Processes and outputs results."""
  percentiles = [50., 80., 90., 95., 99., 99.9]
  buckets = np.percentile(result_list, percentiles).tolist()
  buckets_str = ",".join(
      ["{}:{:.4f}".format(p, b) for p, b in zip(percentiles, buckets)])

  if result_dict["total"] == 0:
    result_dict["total"] = len(result_list)

  # this is what we record for each run
  result = {
      "mean": np.mean(result_list),
      "took": took,
      "count": len(result_list),
      "percentiles": {str(k): v for k, v in zip(percentiles, buckets)},
      "good_items": result_dict["good"],
      "total_items": result_dict["total"],
      "accuracy": 100. * result_dict["good"] / result_dict["total"],
  }

  # add the result to the result dict
  final_results[name] = result

  # to stdout
  log.info("%s, mean=%6f, time=%2f, accuracy=%2f, queries=%d, tiles=%s",
           name, result["mean"], took, result["accuracy"], len(result_list),
           buckets_str)

