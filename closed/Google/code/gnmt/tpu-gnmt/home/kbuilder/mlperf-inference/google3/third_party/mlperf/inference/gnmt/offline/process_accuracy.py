"""Utility function to calculate accuracy for NMT."""

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

import argparse
import codecs
import os
import sys
from scripts import bleu
from utils import evaluation_utils as e_utils


def get_accuracy(reference, accuracy_log):
  """Retuns the bleu score with given reference and accuracy log file."""
  ##
  # @note: List of lists of words from the reference
  # @note: ref[i][j] refers to the j'th word of sentence i
  ref = []
  with codecs.getreader("utf-8")(open(reference, "rb")) as ifh:
    ref_sentences = ifh.readlines()
    # Sanitize each sentence and convert to array of words
    ref = [e_utils._clean(s, "bpe").split(" ") for s in ref_sentences]  # pylint: disable=protected-access

  running_blue = bleu.RunningBLEUScorer(4)

  seen_sentence_ids = set()
  with open(accuracy_log) as ifh:
    for line in ifh:
      ##
      # @note: Simplistic way to extract records
      # @note: This could be loaded in through json module,
      # but it is an easy way to test incomplete log files
      # without proper closure braces.
      if "{" not in line:
        continue
      s_line = line.strip("\n").strip(",")
      record = eval(s_line)  # pylint: disable=eval-used

      # Decode data to sentence
      sentence = (bytes.fromhex(record["data"])).decode("utf-8")
      trans = sentence.split(" ")
      sent_id = record["qsl_idx"]

      # Skip duplicates
      if sent_id in seen_sentence_ids:
        continue

      # Keep track of the sentence IDs seen before
      seen_sentence_ids.add(sent_id)

      # Update the Running BLEU Scorer for this sentence
      running_blue.add_sentence(ref[sent_id], trans)

  (accuracy, _, _, _, _, _) = running_blue.calc_bleu_score()

  return accuracy


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--reference",
      type=str,
      default=os.path.join(os.getcwd(), "nmt", "data",
                           "newstest2014.tok.bpe.32000.de"),
      help="Reference text to compare accuracy against.")
  parser.add_argument(
      "--accuracy_log",
      type=str,
      default="mlperf_log_accuracy.json",
      help="Accuracy log file")
  args = parser.parse_args()

  # Check whether reference and log files exist
  if not os.path.exists(args.reference):
    print(
        "Could not find reference file {}. Please specify its location".format(
            args.reference))
    sys.exit(0)

  if not os.path.exists(args.accuracy_log):
    print("Could not find accuracy log file {}. Please specify its location"
          .format(args.accuracy_log))
    sys.exit(0)

  blue = get_accuracy(args.reference, args.accuracy_log)
  print("BLEU: %.2f" % (bleu * 100))  # pylint: disable=superfluous-parens
