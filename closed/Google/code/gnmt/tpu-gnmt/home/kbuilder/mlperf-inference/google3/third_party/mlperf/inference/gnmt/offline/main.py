# Copyright 2018 The MLPerf Authors. All Rights Reserved.
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
# =============================================================================
"""GNMT MLPerf inference benchmark."""

from __future__ import print_function

import gc
import os
import tempfile
from absl import app
from absl import flags
import tensorflow as tf
import generic_loadgen
import loadgen_gnmt
import process_accuracy
import mlperf_loadgen

flags.DEFINE_string(
    "scenario",
    default="Offline",
    help="Scenario to be run: can be one of {Server, Offline}")

flags.DEFINE_integer(
    "batch_size",
    default=64,
    help="Max batch size to use in Offline and MultiStream scenarios.")

flags.DEFINE_integer(
    "batch_timeout_micros", default=80000, help="batch timeout in micros")

flags.DEFINE_integer("time", default=60, help="time to scan in seconds")

flags.DEFINE_integer("query_count", default=3003, help="query count")

flags.DEFINE_bool(
    "store_translation",
    default=False,
    help="Store the output of translation? Note: Only valid with SingleStream scenario."
)

flags.DEFINE_string(
    "input_file",
    default="/nmt/inference_data/data/newstest2014.tok.bpe.32000.en.large",
    help="path to the dataset")

flags.DEFINE_string(
    "ckpt_path",
    default="/nmt/translate.ckpt",
    help="path to the checkpoint")

flags.DEFINE_string(
    "vocab_prefix",
    default="/nmt/inference_data/data/vocab.bpe.32000",
    help="path to the checkpoint")

flags.DEFINE_string(
    "hparams_path",
    default="/nmt/wmt16_gnmt_4_layer.json",
    help="path to the hparams path")

flags.DEFINE_string(
    "outdir",
    default="/test/nmt",
    help="path to the output")

flags.DEFINE_bool("verbose", default=False, help="Verbose output.")

flags.DEFINE_string("master", default="", help="path to the output")

flags.DEFINE_integer("qps", default=2700, help="target qps estimate")

flags.DEFINE_bool("accuracy_mode", default=False, help="Accuracy mode.")

flags.DEFINE_string(
    "reference",
    default="/nmt/newstest2014.tok.bpe.32000.de",
    help="path to the reference output file")

flags.DEFINE_integer(
    "qsl_rng_seed", default=0x2b7e151628aed2a6, help="QSL rng seed.")

flags.DEFINE_integer(
    "sample_index_rng_seed",
    default=0x093c467e37db0c7a,
    help="Sample index rng seed.")

flags.DEFINE_integer(
    "schedule_rng_seed", default=0x3243f6a8885a308d, help="Schedule rng seed.")

FLAGS = flags.FLAGS

MILLI_SEC = 1000
NANO_SEC = 1e9

SCENARIO_MAP = {
    "SingleStream": mlperf_loadgen.TestScenario.SingleStream,
    "MultiStream": mlperf_loadgen.TestScenario.MultiStream,
    "Server": mlperf_loadgen.TestScenario.Server,
    "Offline": mlperf_loadgen.TestScenario.Offline,
}


def flush_queries():
  pass


def main(argv):
  del argv

  settings = mlperf_loadgen.TestSettings()
  settings.qsl_rng_seed = FLAGS.qsl_rng_seed
  settings.sample_index_rng_seed = FLAGS.sample_index_rng_seed
  settings.schedule_rng_seed = FLAGS.schedule_rng_seed
  if FLAGS.accuracy_mode:
    settings.mode = mlperf_loadgen.TestMode.AccuracyOnly
  else:
    settings.mode = mlperf_loadgen.TestMode.PerformanceOnly
  settings.scenario = SCENARIO_MAP[FLAGS.scenario]
  if FLAGS.qps:
    qps = float(FLAGS.qps)
    settings.server_target_qps = qps
    settings.offline_expected_qps = qps

  if FLAGS.scenario == "Offline" or FLAGS.scenario == "Server":
    masters = FLAGS.master
    masters = masters.split(",")
    if len(masters) < 1:
      masters = [FLAGS.master]

    runner = loadgen_gnmt.GNMTRunner(
        input_file=FLAGS.input_file,
        ckpt_path=FLAGS.ckpt_path,
        hparams_path=FLAGS.hparams_path,
        vocab_prefix=FLAGS.vocab_prefix,
        outdir=FLAGS.outdir,
        batch_size=FLAGS.batch_size,
        verbose=FLAGS.verbose,
        masters=masters,
        scenario=FLAGS.scenario)

    runner.load(FLAGS.batch_timeout_micros)

    # Specify exactly how many queries need to be made
    settings.min_query_count = FLAGS.qps * FLAGS.time
    settings.max_query_count = 0
    settings.min_duration_ms = 60 * MILLI_SEC
    settings.max_duration_ms = 0
    settings.server_target_latency_ns = int(0.25 * NANO_SEC)
    settings.server_target_latency_percentile = 0.97

  else:
    print("Invalid scenario selected")
    assert False

  # Create a thread in the GNMTRunner to start accepting work
  runner.start_worker()

  # Maximum sample ID + 1
  total_queries = FLAGS.query_count
  # Select the same subset of $perf_queries samples
  perf_queries = FLAGS.query_count

  sut = mlperf_loadgen.ConstructSUT(runner.enqueue, flush_queries,
                                    generic_loadgen.process_latencies)
  qsl = mlperf_loadgen.ConstructQSL(total_queries, perf_queries,
                                    runner.load_samples_to_ram,
                                    runner.unload_samples_from_ram)

  log_settings = mlperf_loadgen.LogSettings()
  log_settings.log_output.outdir = tempfile.mkdtemp()
  # Disable detail logs to prevent it from stepping on the summary
  # log in stdout on some systems.
  log_settings.log_output.copy_detail_to_stdout = False
  log_settings.log_output.copy_summary_to_stdout = True
  log_settings.enable_trace = False
  mlperf_loadgen.StartTestWithLogSettings(sut, qsl, settings, log_settings)

  runner.finish()
  mlperf_loadgen.DestroyQSL(qsl)
  mlperf_loadgen.DestroySUT(sut)

  for oldfile in tf.gfile.Glob(
      os.path.join(log_settings.log_output.outdir, "*")):
    basename = os.path.basename(oldfile)
    newfile = os.path.join(FLAGS.outdir, basename)
    tf.gfile.Copy(oldfile, newfile, overwrite=True)

  if FLAGS.accuracy_mode:
    log_accuracy = os.path.join(log_settings.log_output.outdir,
                                "mlperf_log_accuracy.json")
    tf.gfile.Copy(FLAGS.reference, "/tmp/reference")
    bleu = process_accuracy.get_accuracy("/tmp/reference", log_accuracy)
    print("BLEU: %.2f" % (bleu * 100))  # pylint: disable=superfluous-parens


if __name__ == "__main__":
  # Disable garbage collection to avoid stalls when running tests.
  gc.disable()
  app.run(main)
