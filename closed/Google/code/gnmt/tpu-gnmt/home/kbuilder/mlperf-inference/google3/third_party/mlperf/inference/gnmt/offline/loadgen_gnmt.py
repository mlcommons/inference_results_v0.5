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
"""GNMT runner."""
from __future__ import print_function

import argparse
import array
import os
import threading
import time
import numpy as np
import tensorflow as tf

from tpu import nmt
from tpu.utils import iterator_utils
from tpu.utils import vocab_utils
from utils import nmt_utils
import backend_tf
import generic_loadgen
import mlperf_loadgen


class TranslationTask(object):

  def __init__(self, query_id, sentence_id, output_file):
    self.query_id = [query_id]
    self.sentence_id = sentence_id
    self.output_file = output_file
    self.start = time.time()


class BatchTranslationTask(object):

  def __init__(self, sentence_id_list, query_id_list):
    self.sentence_id_list = sentence_id_list
    self.query_id_list = query_id_list
    self.query_id = query_id_list[0]
    self.start = time.time()


##
# @brief Wrapper around TF GNMT Inference that can interface with loadgen
class GNMTRunner(generic_loadgen.Runner):
  """GNMT runner implementation."""
  ##
  # @brief Constructor will build the graph and set some wrapper variables
  # @param input_file: path to the input text
  # @param ckpt_path: path to the GNMT checkpoint
  # @param hparams_path: path to the parameters used to configure GNMT graph
  # @param vocab_prefix: Path to vocabulary file (don't add .en or .de suffixes)
  # @param outdir: Output directory to optionally write translations to
  # @param batch_size: batch size to use when processing BatchTranslationTasks

  def __init__(self,
               input_file=None,
               ckpt_path=None,
               hparams_path=None,
               vocab_prefix=None,
               outdir=None,
               batch_size=32,
               verbose=False,
               masters=None,
               scenario="Offline"):
    generic_loadgen.Runner.__init__(self, qSize=0)

    self.scenario = scenario

    # If no value is provided for the construtor arguments, set defaults here
    if input_file is None:
      input_file = os.path.join(os.getcwd(), "nmt", "data",
                                "newstest2014.tok.bpe.32000.en.large")

    if ckpt_path is None:
      ckpt_path = os.path.join(os.getcwd(), "ende_gnmt_model_4_layer",
                               "translate.ckpt")
    self.ckpt_path = ckpt_path

    if hparams_path is None:
      hparams_path = os.path.join(os.getcwd(), "nmt", "standard_hparams",
                                  "wmt16_gnmt_4_layer.json")

    if vocab_prefix is None:
      vocab_prefix = os.path.join(os.getcwd(), "nmt", "data", "vocab.bpe.32000")

    if outdir is None:
      outdir = os.path.join(os.getcwd(), "lg_output")

    flags = self.parse_options(ckpt_path, hparams_path, vocab_prefix, outdir,
                               batch_size)

    self.backends = []
    self.masters = masters
    if self.masters is None:
      self.masters = [""]
    self.num_tpus = len(self.masters)

    for _ in range(self.num_tpus):
      self.backends.append(
          backend_tf.BackendTensorflow(
              vocab_prefix=vocab_prefix, scenario=self.scenario))

    self.vocab_prefix = vocab_prefix
    self.batch_size = batch_size
    self.graph = tf.Graph()
    self.setup(flags)

    # Wrapper parameters
    self.input_file = input_file
    self.infer_data = []  # This will be filled by load_samples_to_ram
    self.count = 0
    self.VERBOSE = verbose

  ##
  # @brief Parse GNMT-specific options before setting up
  def parse_options(self, ckpt_path, hparams_path, vocab_prefix, outdir,
                    batch_size):
    flags = None
    # TBD remove argument parsing, and just have it return all default values.
    nmt_parser = argparse.ArgumentParser()
    nmt.add_arguments(nmt_parser)
    flags, _ = nmt_parser.parse_known_args()

    # Some of these flags are never used and are just set for consistency
    flags.num_workers = 1
    flags.iterations = 1
    flags.infer_batch_size = batch_size
    flags.num_inter_threads = 1
    flags.num_intra_threads = 1
    flags.run = "accuracy"  # Needs to be set to accuracy to generate output

    # Pass in inference specific flags
    flags.ckpt = ckpt_path
    flags.src = "en"
    flags.tgt = "de"
    flags.hparams_path = hparams_path
    flags.out_dir = outdir
    flags.vocab_prefix = vocab_prefix

    return flags

  ##
  # @brief Configure hparams and setup GNMT graph
  # @pre Requires output from parse_options
  def setup(self, flags):
    # Model output directory
    out_dir = flags.out_dir
    if out_dir and not tf.gfile.Exists(out_dir):
      tf.gfile.MakeDirs(out_dir)

    # Load hparams.
    default_hparams = nmt.create_hparams(flags)
    hparams = nmt.extend_hparams(default_hparams)
    hparams.tgt_sos_id, hparams.tgt_eos_id = 1, 2

    # Inference indices (inference_indices is broken, but without setting it to
    # None we'll crash)
    hparams.inference_indices = None
    hparams.num_gpus = 0

    # Parameters needed by TF GNMT
    self.hparams = hparams
    self.out_dir = out_dir

  def load(self, batch_timeout_micros):
    for tpu_id, master in enumerate(self.masters):
      self.backends[tpu_id].load(
          self.ckpt_path, self.hparams, master, batch_timeout_micros,
          None if self.scenario == "Offline" else [16, 32, 64])

  ##
  # @brief Load sentences into the infer_data array and warmup the network
  def load_samples_to_ram(self, query_samples):
    # We always load the first N samples from the library.
    # Determine if N needs to be the total size of the dataset or not.
    query_samples.sort()
    query_samples_is_first_n = True
    for i, query_sample in enumerate(query_samples):
      if query_sample != i:
        query_samples_is_first_n = False
        break
    first_n_to_load = (
        len(query_samples) if query_samples_is_first_n else 3903900)

    with self.graph.as_default():
      self.sess = tf.Session()
      dataset = tf.data.TextLineDataset(self.input_file)
      vocab_table, _ = vocab_utils.create_vocab_tables(self.vocab_prefix)
      iterator = iterator_utils.get_infer_iterator(
          dataset,
          vocab_table,
          batch_size=first_n_to_load,
          eos=self.hparams.eos,
          src_max_len=self.hparams.src_max_len_infer
      ).make_initializable_iterator()
      self.sess.run(tf.tables_initializer())
      self.sess.run(iterator.initializer)
      r = self.sess.run(iterator.get_next())
    self.sources = r["source"]
    self.seq_lens = r["source_sequence_length"]

    # Warmup
    def init_fn(tpu_id):
      self.backends[tpu_id].warmup()

    threads = []
    for i in range(self.num_tpus):
      thread = threading.Thread(target=init_fn, args=(i,))
      threads.append(thread)
      thread.start()

    for thread in threads:
      thread.join()

    # After warmup, give the system a moment to quiesce before putting it under
    # load.
    time.sleep(5)

  ##
  # @brief Run translation on a number of sentence id's
  # @param sentence_id_list: List of sentence numbers to translate
  # @return Translated sentences
  def translate(self, sentence_id_list, tpu_id=0):
    nmt_outputs = self.backends[tpu_id].predict(
        np.take(self.sources, sentence_id_list, 0),
        np.take(self.seq_lens, sentence_id_list, 0))[0]

    batch_size = nmt_outputs.shape[0]

    translation = []
    for decoded_id in range(batch_size):
      translation += [
          nmt_utils.get_translation(
              nmt_outputs,
              decoded_id,
              tgt_eos=self.hparams.eos,
              subword_option=self.hparams.subword_option)
      ]

    # Keeping track of how many translations happened
    self.count += len(translation)

    return translation

  ##
  # @brief start worker thread
  def start_worker(self):
    self.workers = []
    handler = self.handle_tasks
    threads = os.cpu_count() * 3
    for i in range(threads):
      worker = threading.Thread(target=handler, args=(i % self.num_tpus,))
      worker.daemon = True
      self.workers.append(worker)
      worker.start()

  ##
  # @brief infinite loop that pulls translation tasks from a queue
  # @note This needs to be run by a worker thread
  def handle_tasks(self, tpu_id):
    while True:
      # Block until an item becomes available
      qitem = self.tasks.get(block=True)

      # When a "None" item was added, it is a
      # signal from the parent to indicate we should stop
      # working (see finish)
      if qitem is None:
        break

      results = self.process(qitem, tpu_id)
      response = []
      gc_hack = []
      for res, q_id in zip(results, qitem.query_id_list):
        result_arr = array.array("B", res)
        gc_hack.append(result_arr)
        r_info = result_arr.buffer_info()
        response.append(
            mlperf_loadgen.QuerySampleResponse(q_id, r_info[0], r_info[1]))

      # Tell loadgen that we're ready with this query
      mlperf_loadgen.QuerySamplesComplete(response)

      self.tasks.task_done()

  ##
  # @brief Invoke GNMT to translate the input file
  # @pre Ensure load_samples_to_ram was called to fill self.infer_data
  def process(self, qitem, tpu_id):
    for i in [1]:
      cur_sentid_list = [index for index in qitem.sentence_id_list]
      num_ids = len(cur_sentid_list)
      bs = self.hparams.infer_batch_size if self.scenario == "Offline" else 1
      if num_ids < bs:
        cur_sentid_list += [0] * (bs - num_ids)
      translation = self.translate(cur_sentid_list, tpu_id)[:num_ids]

      if self.VERBOSE:
        print("Performed {} translations".format(self.count))

      return translation

  ##
  # @brief Create a batched task and add it to the queue
  def enqueue(self, query_samples):
    for i in [1]:
      if self.VERBOSE:
        print("Received query")
      sorted_samples = [(self.seq_lens[sample.index], sample.id, sample.index)
                        for sample in query_samples]
      sorted_samples = sorted(sorted_samples)
      query_id_list = [sample[1] for sample in sorted_samples]
      sentence_id_list = [sample[2] for sample in sorted_samples]

      bs = self.hparams.infer_batch_size
      num_samples = len(query_samples)
      for i in range(0, num_samples, bs):
        task = BatchTranslationTask(sentence_id_list[i:i + bs],
                                    query_id_list[i:i + bs])
        self.tasks.put(task)
