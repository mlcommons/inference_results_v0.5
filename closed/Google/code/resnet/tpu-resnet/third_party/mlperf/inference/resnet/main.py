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
"""mlperf inference ResNet-50 benchmark."""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import array
import gc
import json
import logging
import os
import sys
import tempfile
import threading
import time
import traceback

from absl import app
from absl import flags

import numpy as np
from six.moves import queue
import tensorflow as tf

import mlperf_loadgen as lg

import utils
from model import backend_tf

BackendTensorflow = backend_tf.BackendTensorflow

flags.DEFINE_string("dataset", default="imagenet", help="dataset")
flags.DEFINE_string(
    "dataset_path",
    default=None,
    help="path to the dataset")
flags.DEFINE_string(
    "dataset_list", default=None, help="path to the dataset list")
flags.DEFINE_enum(
    "data_format",
    default="NHWC",
    enum_values=["NCHW", "NHWC"],
    help="data format")
flags.DEFINE_enum(
    "profile", default=None, enum_values=["defaults", "resnet50-tf"],
    help="standard profiles")
flags.DEFINE_enum(
    "scenario",
    default="SingleStream",
    enum_values=["SingleStream", "MultiSteam", "Server", "Offline"],
    help="benchmark scenario.")
flags.DEFINE_bool(
    "preprocessing_and_graph_only", default=False, help="When enabled, the "
    "program runs preprocessing and graph generation; it doesn't run the test.")
flags.DEFINE_string(
    "model",
    default="/resnet/resnet_imagenet_v1_fp32_20181001/model.ckpt-225207",
    help="model file")
flags.DEFINE_string("outdir", default=None, help="directory for test results")
flags.DEFINE_string(
    "preprocess_outdir", default=None, help="directory for test results")
flags.DEFINE_string(
    "export_outdir", default=None, help="directory for exporting model")
flags.DEFINE_integer(
    "threads", default=os.cpu_count(), help="number of threads")
flags.DEFINE_integer("time", default=None, help="time to scan in seconds")
flags.DEFINE_integer("total_sample_count", default=0,
                     help="dataset items to use")
flags.DEFINE_multi_integer("batch_size", default=None,
                           help="a list of batch size for inference")
flags.DEFINE_integer("qps", default=10, help="target qps estimate")
flags.DEFINE_string(
    "max_latency",
    default=utils.DEFAULT_LATENCY_BUCKETS,
    help="max latency in milliseconds at 99 percent-tile")
flags.DEFINE_bool("cache", default=False, help="use cache")
flags.DEFINE_bool("accuracy", default=False, help="enable accuracy pass")
# Performance settings
flags.DEFINE_integer("space_to_depth_block_size", default=0,
                     help="conv0 space-to-depth block size")
flags.DEFINE_bool("tpu_transpose", default=True,
                  help="To transpose image tensors for TPU performance.")
# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    "tpu_name",
    default="local",
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")
flags.DEFINE_string(
    "gcp_project",
    default=None,
    help="Project name for the Cloud TPU-enabled project. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_string(
    "tpu_zone",
    default=None,
    help="GCE zone where the Cloud TPU is located in. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_integer(
    "num_tpus", default=1, help="number of tpus for inference")

FLAGS = flags.FLAGS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

NANO_SEC = 1e9
MILLI_SEC = 1000

last_timing = []


class Item(object):
  """An item that we queue for processing by the thread pool."""

  def __init__(self, query_id, content_id):
    self.query_id = query_id
    self.content_id = content_id
    self.start = time.time()


class Runner(object):
  """A runner object that handles the inference task."""

  def __init__(self, model, ds, batch_size, threads, post_process=None):
    self.tasks = queue.Queue(maxsize=(1 << 14))
    self.workers = []
    self.model = model
    self.post_process = post_process
    self.threads = threads
    self.result_dict = {}
    self.take_accuracy = False
    self.ds = ds
    self.batch_size = batch_size

  def handle_tasks(self, tasks_queue):
    """Worker thread."""
    while True:
      qitem = tasks_queue.get()
      if qitem is None:
        # None in the queue indicates the parent want us to exit
        tasks_queue.task_done()
        break

      try:
        # run the prediction
        indices, label = self.ds.get_indices(qitem.content_id)
        if len(indices) < min(self.batch_size) and FLAGS.scenario == "Offline":
          indices = np.pad(
              indices, ((0, min(self.batch_size) - len(indices))),
              "constant",
              constant_values=(0, 0))

        results = self.model.predict(indices)
        if self.take_accuracy:
          self.post_process(results, qitem.content_id, label, self.result_dict)
      except Exception as ex:  # pylint: disable=broad-except
        log.fatal("thread: failed with ex=%s %s", ex, traceback.format_exc())
      finally:
        response_array_refs = []
        response = []
        for idx, query_id in enumerate(qitem.query_id):
          bi = [0, 0]
          if self.take_accuracy:
            response_array = array.array(
                "B",
                np.array(results[0][idx], np.float32).tobytes())
            response_array_refs.append(response_array)
            bi = response_array.buffer_info()
          response.append(lg.QuerySampleResponse(query_id, bi[0], bi[1]))
        lg.QuerySamplesComplete(response)
      tasks_queue.task_done()

  def start_pool(self):
    handler = self.handle_tasks
    for _ in range(self.threads):
      worker = threading.Thread(target=handler, args=(self.tasks,))
      worker.daemon = True
      self.workers.append(worker)
      worker.start()

  def start_run(self, result_dict, take_accuracy):
    self.result_dict = result_dict
    self.take_accuracy = take_accuracy
    self.post_process.start()

  def enqueue(self, q_id, c_ids):
    item = Item(q_id, c_ids)
    self.tasks.put(item)

  def finish(self):
    # exit all threads
    for _ in self.workers:
      self.tasks.put(None)
    for worker in self.workers:
      worker.join()

  def warmup(self, sample):
    """Warmup TPU."""
    for batch_size in self.batch_size:
      warmup_indices, _ = self.ds.get_indices([sample] * batch_size)
      for _ in range(32):
        _ = self.model.predict(warmup_indices)

  def get_post_process(self):
    return self.post_process


class MultiCloudTpuRunner(Runner):
  """A MultiCloudTpu runner object that handles the inference task."""

  def __init__(self, models, ds, batch_size, threads, post_process=None):
    super(MultiCloudTpuRunner, self).__init__(None, ds, batch_size, threads,
                                              post_process)
    self.tasks = queue.Queue(maxsize=(1 << 14))
    self.workers = []
    self.models = models
    self.post_process = post_process
    self.threads = threads
    self.result_dict = {}
    self.take_accuracy = False
    self.ds = ds
    self.batch_size = batch_size
    self.num_tpus = len(models)

  def handle_tasks(self, cloud_tpu_id):
    """Worker thread."""
    tasks_queue = self.tasks
    while True:
      qitem = tasks_queue.get()
      if qitem is None:
        # None in the queue indicates the parent want us to exit
        tasks_queue.task_done()
        break

      try:
        # run the prediction
        indices, label = self.ds.get_indices(qitem.content_id)
        if len(indices) < min(self.batch_size) and FLAGS.scenario == "Offline":
          indices = np.pad(
              indices, ((0, min(self.batch_size) - len(indices))),
              "constant",
              constant_values=(0, 0))

        results = self.models[cloud_tpu_id].predict(indices)
        if self.take_accuracy:
          self.post_process(results, qitem.content_id, label, self.result_dict)
      except Exception as ex:  # pylint: disable=broad-except
        src = [self.ds.get_item_loc(i) for i in qitem.content_id]
        log.error("thread: failed with ex=%s, contentid=%s", ex, src)
      finally:
        response_array_refs = []
        response = []
        for idx, query_id in enumerate(qitem.query_id):
          bi = [0, 0]
          if self.take_accuracy:
            response_array = array.array(
                "B",
                np.array(results[0][idx], np.float32).tobytes())
            response_array_refs.append(response_array)
            bi = response_array.buffer_info()
          response.append(lg.QuerySampleResponse(query_id, bi[0], bi[1]))
        lg.QuerySamplesComplete(response)
      tasks_queue.task_done()

  def start_pool(self):
    handler = self.handle_tasks
    for i in range(self.threads):
      worker = threading.Thread(target=handler, args=(i % self.num_tpus,))
      worker.daemon = True
      self.workers.append(worker)
      worker.start()

  def warmup(self, sample, cloud_tpu_id):
    """Warmup TPU."""
    for batch_size in self.batch_size:
      warmup_indices, _ = self.ds.get_indices([sample] * batch_size)
      for _ in range(32):
        _ = self.models[cloud_tpu_id].predict(warmup_indices)


def setup():
  """Configures dataset, model, backend, and runner."""
  if FLAGS.preprocess_outdir:
    outdir = FLAGS.preprocess_outdir
  elif FLAGS.outdir:
    outdir = FLAGS.outdir
  else:
    outdir = tempfile.mkdtemp()
  backend, ds, post_process, count = utils.setup(
      FLAGS.dataset, FLAGS.dataset_path, FLAGS.dataset_list, FLAGS.cache,
      outdir, FLAGS.space_to_depth_block_size,
      FLAGS.tpu_transpose, FLAGS.data_format, FLAGS.total_sample_count)
  final_results = {
      "runtime": backend.name(),
      "version": backend.version(),
      "time": int(time.time()),
  }

  if FLAGS.num_tpus == 1:
    runner = Runner(
        backend, ds, FLAGS.batch_size, FLAGS.threads, post_process=post_process)
  else:
    backend_lists = []
    for _ in range(FLAGS.num_tpus):
      backend_lists.append(BackendTensorflow())
    runner = MultiCloudTpuRunner(
        backend_lists,
        ds,
        FLAGS.batch_size,
        FLAGS.threads,
        post_process=post_process)
  return final_results, count, runner


def run():
  """Runs the offline mode."""
  global last_timing

  # Initiazation
  final_results, count, runner = setup()

  #
  # run the benchmark with timing
  #
  runner.start_pool()

  def issue_query_offline(query_samples):
    """Adds query to the queue."""
    for i in [1]:
      idx = np.array([q.index for q in query_samples])
      query_id = np.array([q.id for q in query_samples])
      batch_size = FLAGS.batch_size[0]
      for i in range(0, len(query_samples), batch_size):
          runner.enqueue(query_id[i:i + batch_size], idx[i:i + batch_size])

  def flush_queries():
    pass

  def process_latencies(latencies_ns):
    global last_timing
    last_timing = [t / 1e9 for t in latencies_ns]

  sut = lg.ConstructSUT(issue_query_offline, flush_queries, process_latencies)

  masters = []

  outdir = FLAGS.outdir if FLAGS.outdir else tempfile.mkdtemp()
  export_outdir = FLAGS.export_outdir if FLAGS.export_outdir else outdir
  export_outdir = os.path.join(export_outdir, "export_model")

  def load_query_samples(sample_list):
    """Load query samples."""
    runner.ds.load_query_samples(sample_list)
    # Find tpu master.
    if FLAGS.num_tpus == 1:
      runner.model.update_qsl(runner.ds.get_image_list_inmemory())
    else:
      for i in range(FLAGS.num_tpus):
        runner.models[i].update_qsl(runner.ds.get_image_list_inmemory())

  def warmup():
    """Warmup the TPUs."""
    load_query_samples([0])
    if FLAGS.num_tpus == 1:
      log.info("warmup ...")
      runner.warmup(0)
      log.info("warmup done")
    else:
      for cloud_tpu_id in range(FLAGS.num_tpus):
        log.info("warmup %d...", cloud_tpu_id)
        runner.warmup(0, cloud_tpu_id)
        log.info("warmup %d done", cloud_tpu_id)

    # After warmup, give the system a moment to quiesce before putting it under
    # load.
    time.sleep(1)

  if FLAGS.num_tpus == 1:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    master = tpu_cluster_resolver.get_master()

    runner.model.build_and_export(
        FLAGS.model,
        export_model_path=export_outdir,
        batch_size=FLAGS.batch_size,
        master=master,
        scenario=FLAGS.scenario)
    runner.model.load(export_model_path=export_outdir, master=master)
  else:
    # Use the first TPU instance to build and export the graph.
    tpu_names = FLAGS.tpu_name
    tpu_names = tpu_names.split(",")
    for tpu_name in tpu_names:
      tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
          tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
      masters.append(tpu_cluster_resolver.get_master())

    runner.models[0].build_and_export(
        FLAGS.model,
        export_model_path=export_outdir,
        batch_size=FLAGS.batch_size,
        master=masters[0],
        scenario=FLAGS.scenario)

    def init_fn(cloud_tpu_id):
      """Init and warmup each cloud tpu."""
      runner.models[cloud_tpu_id].load(
          export_model_path=export_outdir, master=masters[cloud_tpu_id])

    threads = []
    for i in range(FLAGS.num_tpus):
      thread = threading.Thread(target=init_fn, args=(i,))
      threads.append(thread)
      thread.start()

    for thread in threads:
      thread.join()

  warmup()

  qsl = lg.ConstructQSL(count, min(count, 1024), load_query_samples,
                        runner.ds.unload_query_samples)

  test_scenarios = FLAGS.scenario
  if test_scenarios is None:
    test_scenarios_list = []
  else:
    test_scenarios_list = test_scenarios.split(",")

  max_latency = FLAGS.max_latency
  max_latency_list = max_latency.split(",")
  for scenario in test_scenarios_list:
    for target_latency in max_latency_list:
      log.info("starting %s, latency=%s", scenario, target_latency)
      settings = lg.TestSettings()
      log.info(scenario)
      if FLAGS.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly

      settings.scenario = utils.SCENARIO_MAP[scenario]

      if FLAGS.qps:
        qps = float(FLAGS.qps)
        settings.server_target_qps = qps
        settings.offline_expected_qps = qps

      if FLAGS.time:
        settings.min_duration_ms = 60 * MILLI_SEC
        settings.max_duration_ms = 0
        qps = FLAGS.qps or 100
        settings.min_query_count = qps * FLAGS.time
        settings.max_query_count = int(1.1 * qps * FLAGS.time)
      else:
        settings.min_query_count = (1 << 21)

      if FLAGS.time or FLAGS.qps and FLAGS.accuracy:
        settings.mode = lg.TestMode.PerformanceOnly
      # FIXME: add SubmissionRun once available

      target_latency_ns = int(float(target_latency) * (NANO_SEC / MILLI_SEC))
      settings.single_stream_expected_latency_ns = target_latency_ns
      settings.multi_stream_target_latency_ns = target_latency_ns
      settings.server_target_latency_ns = target_latency_ns

      log_settings = lg.LogSettings()
      # TODO(brianderson): figure out how to use internal file path.
      log_settings.log_output.outdir = tempfile.mkdtemp()
      log_settings.log_output.copy_detail_to_stdout = True
      log_settings.log_output.copy_summary_to_stdout = True
      log_settings.enable_trace = False

      result_dict = {"good": 0, "total": 0, "scenario": str(scenario)}
      runner.start_run(result_dict, FLAGS.accuracy)

      lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)

      if FLAGS.accuracy:
        runner.get_post_process().finalize(result_dict, runner.ds)

      utils.add_results(
          final_results, "{}-{}".format(scenario, target_latency),
          result_dict, last_timing,
          time.time() - runner.ds.last_loaded)

  #
  # write final results
  #
  if FLAGS.outdir:
    outfile = os.path.join(FLAGS.outdir, "results.txt")
    with tf.gfile.Open(outfile, "w") as f:
      json.dump(final_results, f, sort_keys=True, indent=4)
  else:
    json.dump(final_results, sys.stdout, sort_keys=True, indent=4)

  runner.finish()
  lg.DestroyQSL(qsl)
  lg.DestroySUT(sut)


def main(argv):
  del argv

  if FLAGS.scenario == "Offline":
    if len(FLAGS.batch_size) != 1:
      raise ValueError("Offline mode supports only a single batch size.")

  if FLAGS.preprocessing_and_graph_only:
    # The program runs only preprocessing and graph generation.
    _, _, runner = setup()
    outdir = FLAGS.outdir if FLAGS.outdir else tempfile.mkdtemp()
    export_outdir = FLAGS.export_outdir if FLAGS.export_outdir else outdir
    export_outdir = os.path.join(export_outdir, "export_model")
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    master = tpu_cluster_resolver.get_master()
    runner.model.build_and_export(
        FLAGS.model,
        export_model_path=export_outdir,
        batch_size=FLAGS.batch_size,
        master=master,
        scenario=FLAGS.scenario)
  else:
    # Check batch size.
    if not FLAGS.batch_size:
      raise ValueError("batch_size must be set")
    run()


if __name__ == "__main__":
  # Disable garbage collection to avoid stalls when running tests.
  gc.disable()
  app.run(main)
