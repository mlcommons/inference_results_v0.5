#ifndef MLPERF_INFERENCE_RUNNER_RUNNER_H_
#define MLPERF_INFERENCE_RUNNER_RUNNER_H_

#include <stdint.h>

#include <vector>

#include "mlperf/inference/loadgen/query_sample.h"
#include "mlperf/inference/runner/loader.h"
#include "mlperf/inference/runner/options.h"
#include "mlperf/inference/runner/queue.h"
#include "tensorflow/core/framework/tensor.h"

namespace mlperf {

typedef std::vector<QuerySample> QuerySamples;
typedef std::vector<QuerySampleResponse*> QuerySampleResponses;

class Runner {
 public:
  Runner(const mlperf::BatchingOption batching_option, int num_threads,
         const std::string export_model_path = "",
         const std::string tpu_target = "", bool standalone = false,
         bool init_tpu = true);
  void Enqueue(const QuerySamples& samples);
  void StartRun();
  void HandleTask(int thread_id);
  void UpdateQSL(const tensorflow::Tensor& qsl,
                 std::unordered_map<QuerySampleIndex, QuerySampleIndex>
                     sample_id_to_qsl_index_map);
  void UpdateQslIndexMap(std::unordered_map<QuerySampleIndex, QuerySampleIndex>
                             sample_id_to_qsl_index_map);
  void WarmUp();
  void Finish();
  ~Runner() {
    Finish();
    for (int i = 0; i < num_threads_; i++) {
      delete[] responses_[i];
    }
  }

 private:
  const mlperf::BatchingOption batching_option_;
  const int num_threads_;
  const std::string export_model_path_;
  const bool standalone_;
  int num_tpus_;
  Queue<QuerySamples> queue_;
  std::vector<std::thread> workers_;
  QuerySampleResponses responses_;

  std::vector<std::unique_ptr<mlperf::Loader>> loader_;
  std::vector<std::string> tpu_target_;
  std::vector<tensorflow::Tensor> inputs_;
  std::vector<tensorflow::Tensor> inputs_max_batch_size_;
  std::vector<std::vector<tensorflow::Tensor>> outputs_;
  std::unordered_map<QuerySampleIndex, QuerySampleIndex>
      sample_id_to_qsl_index_map_;
};
}  // namespace mlperf

#endif  // MLPERF_INFERENCE_RUNNER_RUNNER_H_
