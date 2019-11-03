#include "mlperf/inference/runner/runner.h"

#include <cstdlib>
#include <random>

#include "mlperf/inference/loadgen/loadgen.h"
#include "mlperf/inference/loadgen/query_sample.h"
#include "mlperf/inference/runner/loader.h"
#include "mlperf/inference/runner/options.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace mlperf {

Runner::Runner(const mlperf::BatchingOption batching_option, int num_threads,
               const std::string export_model_path,
               const std::string tpu_target, bool standalone, bool init_tpu)
    : batching_option_(batching_option),
      num_threads_(num_threads),
      export_model_path_(export_model_path),
      standalone_(standalone) {
  // Parse the tpu_target seprated by ","
  std::string delimiter = ",";
  size_t pos = 0;
  std::string token;
  std::string tpu_target_string = tpu_target;
  while ((pos = tpu_target_string.find(delimiter)) != std::string::npos) {
    token = tpu_target_string.substr(0, pos);
    tpu_target_.push_back(token);
    std::cout << token << std::endl;
    tpu_target_string.erase(0, pos + delimiter.length());
  }
  tpu_target_.push_back(tpu_target_string);

  num_tpus_ = std::max(static_cast<int>(tpu_target_.size()), 1);
  loader_.resize(num_tpus_);

  // Initialized loaders.
  for (int i = 0; i < num_tpus_; i++) {
    loader_[i] = absl::make_unique<mlperf::Loader>(
        /*tpu_target_=*/tpu_target_[i],
        /*saved_model_path=*/export_model_path_, batching_option_, init_tpu);
  }

  // Prepare response vectors, input tensors, and output tensors.
  responses_.resize(num_threads_);
  inputs_.resize(num_threads_);
  inputs_max_batch_size_.resize(num_threads_);
  outputs_.resize(num_threads_);
  for (int i = 0; i < num_threads_; i++) {
    responses_[i] = new QuerySampleResponse[batching_option.max_batch_size];
    // TODO(chiachenc): supports arbitrary inputs.
    inputs_[i] =
        tensorflow::Tensor(tensorflow::DT_INT32, tensorflow::TensorShape({1}));
    inputs_max_batch_size_[i] = tensorflow::Tensor(
        tensorflow::DT_INT32,
        tensorflow::TensorShape({batching_option_.max_batch_size}));
    // Zero-initialize input tensors.
    inputs_[i].vec<int>()(0) = 0;
    for (int j = 0; j < batching_option_.max_batch_size; ++j) {
      inputs_max_batch_size_[i].vec<int>()(j) = 0;
    }
    workers_.emplace_back(std::thread(&Runner::HandleTask, this, i));
  }
}

void Runner::Enqueue(const QuerySamples& sample) {
  queue_.put(sample);
}

void Runner::HandleTask(int thread_id) {
  while (true) {
    auto queries = queue_.get();
    if (queries.empty()) {
      break;
    }
    tensorflow::profiler::TraceMe trace_me([&] { return "Predict"; },
                                           /*level=*/2);
    while (!queries.empty() && queries.back().id == 0) {
      queries.pop_back();
    }
    CHECK_LE(queries.size(), batching_option_.max_batch_size);
    CHECK_LE(thread_id, responses_.size());
    // Currently, the approach uses a one-or-max batch size.
    if (queries.size() > 1) {
      for (int64_t i = 0; i < queries.size(); ++i) {
        inputs_max_batch_size_[thread_id].vec<int>()(i) =
            sample_id_to_qsl_index_map_[queries[i].index];
      }
      loader_[thread_id % num_tpus_]->Predict(inputs_max_batch_size_[thread_id],
                                              outputs_[thread_id]);
      uintptr_t raw_data = reinterpret_cast<uintptr_t>(
          outputs_[thread_id][0].tensor_data().data());

      for (int64_t i = 0; i < queries.size(); ++i) {
        responses_[thread_id][i].id = queries[i].id;
        responses_[thread_id][i].data = raw_data + sizeof(int) * i;
        responses_[thread_id][i].size = sizeof(int);
      }
    }else{
      inputs_[thread_id].vec<int>()(0) =
          sample_id_to_qsl_index_map_[queries[0].index];
      loader_[thread_id % num_tpus_]->Predict(inputs_[thread_id],
                                              outputs_[thread_id]);

      responses_[thread_id][0].id = queries[0].id;
      responses_[thread_id][0].data = reinterpret_cast<uintptr_t>(
          outputs_[thread_id][0].tensor_data().data());
      responses_[thread_id][0].size = sizeof(int);
    }

    if (!standalone_) {
      QuerySamplesComplete(responses_[thread_id], queries.size());
    }
  }
}

void Runner::UpdateQslIndexMap(
    std::unordered_map<QuerySampleIndex, QuerySampleIndex>
    sample_id_to_qsl_index_map) {
  sample_id_to_qsl_index_map_ = std::move(sample_id_to_qsl_index_map);
}

void Runner::UpdateQSL(const tensorflow::Tensor& qsl,
                       std::unordered_map<QuerySampleIndex, QuerySampleIndex>
                           sample_id_to_qsl_index_map) {
  for (int i = 0; i < num_tpus_; i++) {
    loader_[i]->UpdateQSL(qsl);
  }
  UpdateQslIndexMap(sample_id_to_qsl_index_map);
}

void Runner::WarmUp() {
  LOG(INFO) << "Warming up the system.";
  for (auto bs : batching_option_.batch_size) {
    LOG(INFO) << "  batch size: " << bs << ", warmup started.";
    tensorflow::Tensor inputs =
        tensorflow::Tensor(tensorflow::DT_INT32, tensorflow::TensorShape({bs}));
    for (int idx = 0; idx < bs; ++idx) {
      inputs.vec<int>()(idx) = 0;
    }
    for (int j = 0; j < num_tpus_; j++) {
      for (int64_t i = 0; i < 32; ++i) {
        loader_[j]->Predict(inputs, outputs_[0]);
      }
    }
    LOG(INFO) << "  batch size: " << bs << ", warmup done.";
  }
}

void Runner::Finish() {
  for (int i = 0; i < num_threads_; i++) {
    QuerySamples empty_queries;
    queue_.put(empty_queries);
  }

  for (int i = 0; i < num_threads_; i++) {
    if (workers_[i].joinable()) {
      workers_[i].join();
    }
  }
}

}  // namespace mlperf
