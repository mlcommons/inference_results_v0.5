#ifndef MLPERF_INFERENCE_RUNNER_DATASET_H_
#define MLPERF_INFERENCE_RUNNER_DATASET_H_

#include <unordered_map>

#include "mlperf/inference/loadgen/query_sample.h"
#include "tensorflow/core/framework/tensor.h"

namespace mlperf {

using int64 = long long int;

tensorflow::Tensor CreateQSLOrDie(
    const std::string& dataset_path,
    const std::vector<QuerySampleIndex>& sample_ids,
    const std::vector<int64>& shape, tensorflow::DataType type);

std::unordered_map<QuerySampleIndex, QuerySampleIndex>
CreateSampleIdToQSLIndexMap(const std::vector<QuerySampleIndex>& sample_ids);

}  // namespace mlperf

#endif  // MLPERF_INFERENCE_RUNNER_DATASET_H_
