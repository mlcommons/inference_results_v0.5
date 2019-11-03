#ifndef MLPERF_INFERENCE_RUNNER_SYSTEM_UNDER_TEST_TPU_H_
#define MLPERF_INFERENCE_RUNNER_SYSTEM_UNDER_TEST_TPU_H_

#include <iostream>

#include "mlperf/inference/loadgen/system_under_test.h"
#include "mlperf/inference/runner/runner.h"

namespace mlperf {
class SystemUnderTestTpu : public SystemUnderTest {
 public:
  SystemUnderTestTpu(std::string name, int batch_size, Runner* runner)
      : name_(name), batch_size_(batch_size), runner_(runner) {}
  ~SystemUnderTestTpu() override = default;

  const std::string& Name() const override { return name_; }

  void IssueQuery(const std::vector<QuerySample>& samples) override {
    for (size_t start = 0; start < samples.size(); start += batch_size_) {
      auto first = samples.cbegin() + start;
      auto last =
          samples.cbegin() + std::min(start + batch_size_, samples.size());
      QuerySamples samples(first, last);
      while (samples.size() < batch_size_) {
        samples.emplace_back(QuerySample{0, samples[0].index});
      }
      runner_->Enqueue(samples);
    }
  }

  void FlushQueries() override {}

  void ReportLatencyResults(
      const std::vector<QuerySampleLatency>& latencies_ns) override {
    // TODO(wangtao): fill the logic here.
  }

 private:
  const std::string name_;
  const int batch_size_;
  Runner* runner_;  // not owned.
};
}  // namespace mlperf

#endif  // MLPERF_INFERENCE_RUNNER_SYSTEM_UNDER_TEST_TPU_H_
