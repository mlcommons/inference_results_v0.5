#ifndef MLPERF_INFERENCE_RUNNER_QUERY_SAMPLE_LIBRARY_TPU_H_
#define MLPERF_INFERENCE_RUNNER_QUERY_SAMPLE_LIBRARY_TPU_H_

#include <string>

#include "mlperf/inference/loadgen/query_sample_library.h"
#include "mlperf/inference/runner/dataset.h"
#include "mlperf/inference/runner/runner.h"

namespace mlperf {

class QuerySampleLibraryTpu : public QuerySampleLibrary {
 public:
  QuerySampleLibraryTpu(std::string name, mlperf::Runner* runner,
                        std::string dataset_path,
                        const std::vector<int64>& shape,
                        tensorflow::DataType datatype,
                        size_t total_sample_count,
                        size_t performance_sample_count)
      : name_(std::move(name)),
        runner_(runner),
        dataset_path_(dataset_path),
        shape_(shape),
        datatype_(datatype),
        total_sample_count_(total_sample_count),
        performance_sample_count_(performance_sample_count) {}
  ~QuerySampleLibraryTpu() override = default;

  const std::string& Name() const override { return name_; }
  size_t TotalSampleCount() { return total_sample_count_; }
  size_t PerformanceSampleCount() { return performance_sample_count_; }

  void LoadSamplesToRam(const std::vector<QuerySampleIndex>& samples) override {
    auto qsl =
        mlperf::CreateQSLOrDie(dataset_path_, samples, shape_, datatype_);
    auto sample_id_to_qsl_index_map =
        mlperf::CreateSampleIdToQSLIndexMap(samples);
    runner_->UpdateQSL(qsl, sample_id_to_qsl_index_map);
  }
  void UnloadSamplesFromRam(
      const std::vector<QuerySampleIndex>& samples) override {}

 private:
  std::string name_;
  mlperf::Runner* runner_;  // Not owned
  std::string dataset_path_;
  std::vector<int64> shape_;
  tensorflow::DataType datatype_;
  size_t total_sample_count_;
  size_t performance_sample_count_;
};

}  // namespace mlperf

#endif  // MLPERF_INFERENCE_RUNNER_QUERY_SAMPLE_LIBRARY_TPU_H_
