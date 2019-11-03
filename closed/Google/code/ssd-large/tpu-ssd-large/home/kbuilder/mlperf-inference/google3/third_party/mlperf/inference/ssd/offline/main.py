"""mlperf inference benchmarking tool
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import array
import gc
import json
import logging
import os
import tempfile
import threading
import time

from absl import app
from absl import flags
import numpy as np
from six.moves import queue
import tensorflow as tf
import mlperf_loadgen as lg

import accuracy_coco
import cocodataset
import ssd_model
from backend_tf import BackendTensorflow

# the datasets we support
SUPPORTED_DATASETS = {
    "coco": (cocodataset.COCODataset, None, None, {
        "image_size": [1200, 1200, 3]
    }),
}

# pre-defined command line options so simplify things. They are used as defaults
# and can be overwritten from command line
DEFAULT_LATENCY_BUCKETS = "0.010,0.050,0.100,0.200,0.400"

SUPPORTED_PROFILES = {
    "defaults": {
        "dataset": "imagenet",
        "backend": "tensorflow",
        "cache": 0,
        "time": 128,
        "max-latency": DEFAULT_LATENCY_BUCKETS,
    },

    # resnet
    "resnet50-tf": {
        "inputs": "input_tensor:0",
        "outputs": "ArgMax:0",
        "dataset": "imagenet",
        "backend": "tensorflow",
    },
}

SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
    "Accuracy": lg.TestMode.AccuracyOnly,
}

flags.DEFINE_string("dataset", default="coco", help="dataset")
flags.DEFINE_string("dataset_path", default=None, help="path to the dataset")
flags.DEFINE_string("annotation_file", default=None, help="annotation file")
flags.DEFINE_string(
    "dataset_list", default=None, help="path to the dataset list")
flags.DEFINE_enum(
    "data_format",
    default="NCHW",
    enum_values=["NCHW", "NHWC"],
    help="data format")
flags.DEFINE_enum(
    "profile", None, SUPPORTED_PROFILES.keys(), help="standard profiles")
flags.DEFINE_string(
    "scenario",
    default="Offline",
    help="benchmark scenario, list of " + str(list(SCENARIO_MAP.keys())))
# TODO(wangtao): fill in the checkpoint file.
flags.DEFINE_string("model", default=None, help="model file")
flags.DEFINE_string(
    "output_model_dir",
    default=None,
    help="directory for converted model checkpoint")
flags.DEFINE_string("outdir", default=None, help="directory for test results")
flags.DEFINE_integer(
    "threads", default=os.cpu_count(), help="number of threads")
flags.DEFINE_integer("time", default=None, help="time to scan in seconds")
flags.DEFINE_integer("count", default=0, help="dataset items to use")
flags.DEFINE_multi_integer(
    "batch_size", default=128, help="a list of batch size for inference")
flags.DEFINE_integer("qps", default=1, help="target qps estimate")
flags.DEFINE_integer(
    "batch_timeout_micros", default=20000, help="batch timeout in micros")
flags.DEFINE_string(
    "max_latency",
    default=DEFAULT_LATENCY_BUCKETS,
    help="max latency in 99pct tile")
flags.DEFINE_bool("cache", default=False, help="use cache")
flags.DEFINE_bool("accuracy", default=False, help="enable accuracy pass")
flags.DEFINE_bool(
    "use_space_to_depth", default=False, help="use space to depth for conv0")
flags.DEFINE_bool("use_bfloat16", default=True, help="use bfloat16.")
flags.DEFINE_string("cache_dir", default=None, help="path to cache dir")
flags.DEFINE_integer("init_iterations", default=8, help="init iterations")
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
flags.DEFINE_bool("use_fused_bn", default=False, help="use fused bn variables.")
flags.DEFINE_integer(
    "qsl_rng_seed", default=0x2b7e151628aed2a6, help="QSL rng seed.")
flags.DEFINE_integer(
    "sample_index_rng_seed",
    default=0x093c467e37db0c7a,
    help="Sample index rng seed.")
flags.DEFINE_integer(
    "schedule_rng_seed", default=0x3243f6a8885a308d, help="Schedule rng seed.")

FLAGS = flags.FLAGS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

NANO_SEC = 1e9
MILLI_SEC = 1000

last_timing = []

FLAGS = flags.FLAGS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")
tf.compat.v1.enable_control_flow_v2()

NANO_SEC = 1e9
MILLI_SEC = 1000

# pylint: disable=missing-docstring

last_timeing = []


class Item(object):
  """An item that we queue for processing by the thread pool."""

  def __init__(self, query_id, content_id):
    self.query_id = query_id
    self.content_id = content_id
    self.start = time.time()


class RunnerBase(object):

  def __init__(self, model, ds, threads, post_proc=None, max_batchsize=128):
    self.ds = ds
    self.model = model
    self.post_process = post_proc
    self.threads = threads
    self.max_batchsize = max_batchsize
    self.result_timing = []

  def handle_tasks(self, tasks_queue):
    pass

  def start_run(self, result_dict, take_accuracy):
    self.result_dict = result_dict
    self.result_timing = []
    self.take_accuracy = take_accuracy

  def run_one_item(self, qitem):
    # run the prediction
    try:
      for i in [1]:
        data, _ = self.ds.get_indices(qitem.content_id)
      results = self.model.predict(data)
    except Exception as ex:  # pylint: disable=broad-except
      src = [self.ds.get_item_loc(i) for i in qitem.content_id]
      log.error("thread: failed on contentid=%s, %s", src, ex)
      results = [[]] * len(qitem.query_id)
      # since post_process will not run, fake empty responses
    finally:
      response = []
      response_arrays = []
      results = np.array(results).reshape((self.max_batchsize, -1)).tolist()
      for idx, query_id in enumerate(qitem.query_id):
        # Ignore padded samples.
        if query_id == -1:
          continue
        response_array = array.array(
            "B",
            np.array(results[idx], np.float32).tobytes())
        response_arrays.append(response_array)
        index = len(response_arrays) - 1
        bi = response_arrays[index].buffer_info()
        response.append(lg.QuerySampleResponse(query_id, bi[0], bi[1]))
      lg.QuerySamplesComplete(response)

  def enqueue(self, query_samples):
    idx = [q.index for q in query_samples]
    query_id = [q.id for q in query_samples]
    # TODO(wangtao): find another proper way to do padding.
    if len(query_samples) < self.max_batchsize:
      padding = self.max_batchsize - len(idx)
      padding_value = 0 if idx is None else idx[0]
      idx = idx.extend([padding_value] * padding)
      for i in [1]:
        self.run_one_item(Item(query_id, idx))
    else:
      bs = self.max_batchsize
      for i in range(0, len(idx), bs):
        for i in [1]:
          self.run_one_item(
              Item(query_id[i:i + bs], idx[i:i + bs]))

  def finish(self):
    pass


class QueueRunner(RunnerBase):

  def __init__(self, models, ds, threads, post_proc=None, max_batchsize=128):
    super(QueueRunner, self).__init__(None, ds, threads, post_proc,
                                      max_batchsize)
    self.tasks = queue.Queue(maxsize=0)
    self.workers = []
    self.result_dict = {}
    self.models = models
    self.num_cloud_tpus = len(models)

    for i in range(self.threads):
      worker = threading.Thread(
          target=self.handle_tasks, args=(i % self.num_cloud_tpus,))
      worker.daemon = True
      self.workers.append(worker)
      worker.start()

  def run_one_item(self, qitem, cloud_tpu_id):
    # run the prediction
    try:
      for i in [1]:
        data, _ = self.ds.get_indices(qitem.content_id)
        results = self.models[cloud_tpu_id].predict(data)
    except Exception as ex:  # pylint: disable=broad-except
      src = [self.ds.get_item_loc(i) for i in qitem.content_id]
      log.error("thread: failed on contentid=%s, %s", src, ex)
      results = [[]] * len(qitem.query_id)
      # since post_process will not run, fake empty responses
    finally:
      response = []
      response_arrays = []
      # Ignore padded samples.
      results = np.array(results).reshape((self.max_batchsize, -1)).tolist()
      for idx, query_id in enumerate(qitem.query_id):
        if query_id == -1:
          continue
        response_array = array.array(
            "B",
            np.array(results[idx], np.float32).tobytes())
        response_arrays.append(response_array)
        index = len(response_arrays) - 1
        bi = response_arrays[index].buffer_info()
        response.append(lg.QuerySampleResponse(query_id, bi[0], bi[1]))
      lg.QuerySamplesComplete(response)

  def handle_tasks(self, cloud_tpu_id):
    """Worker thread."""
    tasks_queue = self.tasks
    while True:
      qitem = tasks_queue.get()
      if qitem is None:
        # None in the queue indicates the parent want us to exit
        tasks_queue.task_done()
        break
      for i in [1]:
        self.run_one_item(qitem, cloud_tpu_id)
      tasks_queue.task_done()

  def enqueue(self, query_samples):
    idx = [q.index for q in query_samples]
    query_id = [q.id for q in query_samples]
    # TODO(wangtao): find another proper way to do padding.
    if len(query_samples) % self.max_batchsize > 0:
      padding = self.max_batchsize - len(idx) % self.max_batchsize
      idx.extend([idx[0]] * padding)
      query_id.extend([-1] * padding)

    bs = self.max_batchsize
    for i in range(0, len(idx), bs):
      ie = i + bs
      self.tasks.put(Item(query_id[i:ie], idx[i:ie]))

  def finish(self):
    # exit all threads
    for _ in self.workers:
      self.tasks.put(None)
    for worker in self.workers:
      worker.join()


def add_results(final_results, name, result_dict, result_list, took):
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
      "qps": len(result_list) / took,
      "count": len(result_list),
      "percentiles": {str(k): v for k, v in zip(percentiles, buckets)},
      "mAP": result_dict["mAP"]
  }

  # add the result to the result dict
  final_results[name] = result

  # to stdout
  print("{} qps={:.2f}, mean={:.6f}, time={:.2f}, queries={}, tiles={}".format(
      name, result["qps"], result["mean"], took, result["mAP"],
      len(result_list), buckets_str))


def main(argv):
  del argv

  global last_timeing

  if FLAGS.scenario == "Server":
    # Disable garbage collection for realtime performance.
    gc.disable()

  # define backend
  backend = BackendTensorflow()

  # override image format if given
  image_format = FLAGS.data_format if FLAGS.data_format else backend.image_format(
  )

  # dataset to use
  wanted_dataset, pre_proc, post_proc, kwargs = SUPPORTED_DATASETS[
      FLAGS.dataset]
  ds = wanted_dataset(
      data_path=FLAGS.dataset_path,
      image_list=FLAGS.dataset_list,
      name=FLAGS.dataset,
      image_format=image_format,
      use_cache=FLAGS.cache,
      count=FLAGS.count,
      cache_dir=FLAGS.cache_dir,
      annotation_file=FLAGS.annotation_file,
      use_space_to_depth=FLAGS.use_space_to_depth)
  # load model to backend
  # TODO(wangtao): parse flags to params.
  params = dict(ssd_model.default_hparams().values())
  params["conv0_space_to_depth"] = FLAGS.use_space_to_depth
  params["use_bfloat16"] = FLAGS.use_bfloat16
  params["use_fused_bn"] = FLAGS.use_fused_bn

  masters = []
  tpu_names = FLAGS.tpu_name
  tpu_names = tpu_names.split(",")
  for tpu_name in tpu_names:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    masters.append(tpu_cluster_resolver.get_master())

  #
  # make one pass over the dataset to validate accuracy
  #
  count = FLAGS.count if FLAGS.count else ds.get_item_count()

  #
  # warmup
  #
  log.info("warmup ...")

  batch_size = FLAGS.batch_size[0] if FLAGS.scenario == "Offline" else 1
  backend_lists = []
  for _ in range(len(tpu_names)):
    backend = BackendTensorflow()
    backend_lists.append(backend)
  runner = QueueRunner(
      backend_lists,
      ds,
      FLAGS.threads,
      post_proc=post_proc,
      max_batchsize=batch_size)

  runner.start_run({}, FLAGS.accuracy)

  def issue_queries(query_samples):
    for i in [1]:
      runner.enqueue(query_samples)

  def flush_queries():
    pass

  def process_latencies(latencies_ns):
    # called by loadgen to show us the recorded latencies
    global last_timeing
    last_timeing = [t / NANO_SEC for t in latencies_ns]

  tf.logging.info("starting {}, latency={}".format(FLAGS.scenario,
                                                   FLAGS.max_latency))
  settings = lg.TestSettings()
  tf.logging.info(FLAGS.scenario)
  settings.scenario = SCENARIO_MAP[FLAGS.scenario]
  settings.qsl_rng_seed = FLAGS.qsl_rng_seed
  settings.sample_index_rng_seed = FLAGS.sample_index_rng_seed
  settings.schedule_rng_seed = FLAGS.schedule_rng_seed

  if FLAGS.accuracy:
    settings.mode = lg.TestMode.AccuracyOnly
  else:
    settings.mode = lg.TestMode.PerformanceOnly

  if FLAGS.qps:
    qps = float(FLAGS.qps)
    settings.server_target_qps = qps
    settings.offline_expected_qps = qps

  if FLAGS.time:
    settings.min_duration_ms = FLAGS.time * MILLI_SEC
    settings.max_duration_ms = 0
    qps = FLAGS.qps or 100
    settings.min_query_count = qps * FLAGS.time
    settings.max_query_count = 0
  else:
    settings.min_query_count = 270336
    settings.max_query_count = 0

  target_latency_ns = int(float(FLAGS.max_latency) * NANO_SEC)
  settings.single_stream_expected_latency_ns = target_latency_ns
  settings.multi_stream_target_latency_ns = target_latency_ns
  settings.server_target_latency_ns = target_latency_ns

  log_settings = lg.LogSettings()
  log_settings.log_output.outdir = tempfile.mkdtemp()
  log_settings.log_output.copy_detail_to_stdout = True
  log_settings.log_output.copy_summary_to_stdout = True
  log_settings.enable_trace = False

  def load_query_samples(sample_list):
    """Load query samples and warmup the model."""
    ds.load_query_samples(sample_list)
    data = ds.get_image_list_inmemory()

    def init_fn(cloud_tpu_id):
      tf.logging.info("Load model for %dth cloud tpu", cloud_tpu_id)
      runner.models[cloud_tpu_id].load(
          FLAGS.model,
          FLAGS.output_model_dir,
          data,
          params,
          batch_size=FLAGS.batch_size,
          master=masters[cloud_tpu_id],
          scenario=FLAGS.scenario,
          batch_timeout_micros=FLAGS.batch_timeout_micros)

      # Init TPU.
      for it in range(FLAGS.init_iterations):
        tf.logging.info("Initialize cloud tpu at iteration %d", it)
        for batch_size in FLAGS.batch_size:
          example, _ = ds.get_indices([sample_list[0]] * batch_size)
          _ = runner.models[cloud_tpu_id].predict(example)

    threads = []
    for i in range(len(tpu_names)):
      thread = threading.Thread(target=init_fn, args=(i,))
      threads.append(thread)
      thread.start()

    for thread in threads:
      thread.join()

  sut = lg.ConstructSUT(issue_queries, flush_queries, process_latencies)
  qsl = lg.ConstructQSL(count, min(count, 350), load_query_samples,
                        ds.unload_query_samples)

  lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)

  runner.finish()
  lg.DestroyQSL(qsl)
  lg.DestroySUT(sut)

  tf.io.gfile.mkdir(FLAGS.outdir)

  for oldfile in tf.gfile.Glob(
      os.path.join(log_settings.log_output.outdir, "*")):
    basename = os.path.basename(oldfile)
    newfile = os.path.join(FLAGS.outdir, basename)
    tf.gfile.Copy(oldfile, newfile, overwrite=True)

  if FLAGS.accuracy:
    with tf.gfile.Open(os.path.join(FLAGS.outdir, "results.txt"), "w") as f:
      results = {"mAP": accuracy_coco.main()}
      json.dump(results, f, sort_keys=True, indent=4)


if __name__ == "__main__":
  app.run(main)
