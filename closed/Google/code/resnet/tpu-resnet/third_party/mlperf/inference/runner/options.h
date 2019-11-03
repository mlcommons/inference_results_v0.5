#ifndef MLPERF_INFERENCE_RUNNER_OPTIONS_H_
#define MLPERF_INFERENCE_RUNNER_OPTIONS_H_

#include <vector>

#include "mlperf/inference/loadgen/test_settings.h"
#include "tensorflow/core/platform/logging.h"

namespace mlperf {

struct BatchingOption {
  std::string thread_pool_name;
  int num_batch_threads;
  int batch_timeout_micros;
  int max_enqueued_batches;
  std::vector<int> batch_size;
  // max_batch_size is derived from batch_size.
  int max_batch_size;
  BatchingOption(
      std::string thread_pool_name = std::string("shared_batch_scheduler"),
      const int num_batch_threads = 16, const int batch_timeout_micros = 1000,
      const int max_enqueued_batches = 1 << 16,
      std::vector<int> batch_size_list = std::vector<int>({4}))
      : thread_pool_name(std::move(thread_pool_name)),
        num_batch_threads(num_batch_threads),
        batch_timeout_micros(batch_timeout_micros),
        max_enqueued_batches(max_enqueued_batches),
        batch_size(std::move(batch_size_list)) {
    max_batch_size = *std::max_element(batch_size.begin(),  batch_size.end());
  }
};

struct Option {
  int num_worker_threads;
  mlperf::TestScenario scenario;
  BatchingOption batching_option;
  int total_sample_count;
  int performance_sample_count;
  int qps;
  int time;
  int max_latency;
  std::string outdir;
  std::string export_model_path;
  std::string tpu_target;
  bool init_tpu;
  Option(const int num_worker_threads = 64,
         const BatchingOption batching_option = BatchingOption(),
         const int total_sample_count = 50000,
         const int performance_sample_count = 3072, const int qps = 2000,
         const int time = 60, const int max_latency = 15,
         std::string outdir = "/tmp", std::string export_model_path = "",
         std::string tpu_target = "", std::string test_scenario = "Server",
         bool init_tpu = true)
      : num_worker_threads(num_worker_threads),
        batching_option(batching_option),
        total_sample_count(total_sample_count),
        performance_sample_count(performance_sample_count),
        qps(qps),
        time(time),
        max_latency(max_latency),
        outdir(std::move(outdir)),
        export_model_path(std::move(export_model_path)),
        tpu_target(std::move(tpu_target)),
        init_tpu(init_tpu) {
    if (test_scenario == "Server") {
      scenario = mlperf::TestScenario::Server;
    } else if (test_scenario == "Offline") {
      scenario = mlperf::TestScenario::Offline;
    } else {
      LOG(ERROR) << "Unsupported scenario.";
    }
  }
};
}  // namespace mlperf
#endif  // MLPERF_INFERENCE_RUNNER_OPTIONS_H_
