#include "mlperf/inference/runner/loader.h"

#include <iterator>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow_serving/batching/batching_session.h"

namespace mlperf {

namespace {
// A GraphDef containing the ops required to initialize and shutdown a TPU.
// This proto was generated from the script oneoffs/generate_tpu_graph_def.py.
constexpr auto kTpuOpsGraphDef = R"(
node {
  name: "ConfigureDistributedTPU"
  op: "ConfigureDistributedTPU"
  device: "/device:TPU_SYSTEM:0"
  attr {
    key: "embedding_config"
    value {
      s: ""
    }
  }
  attr {
    key: "is_global_init"
    value {
      b: false
    }
  }
  attr {
    key: "tpu_embedding_config"
    value {
      s: ""
    }
  }
}
node {
  name: "ShutdownDistributedTPU"
  op: "ShutdownDistributedTPU"
  device: "/device:TPU_SYSTEM:0"
}
library {
}
)";
}  // namespace

Loader::Loader(
    const std::string tpu_target, const std::string saved_model_path,
    const mlperf::BatchingOption batching_options = mlperf::BatchingOption(),
    bool init_tpu = true)
    : tpu_target_(tpu_target),
      saved_model_path_(saved_model_path),
      batching_options_(batching_options) {
  // The initialization order must be the following: (1) load the graph. (2)
  // initialize the TPU system, and (3) create a batching session.
  if (init_tpu) {
    TF_CHECK_OK(InitializeTpu());
  }
  TF_CHECK_OK(LoadSavedModel());
  TF_CHECK_OK(CreateBatchingSession());
}

tensorflow::Status Loader::LoadSavedModel() {
  tensorflow::RunOptions run_options;
  std::unordered_set<std::string> tags = {tensorflow::kSavedModelTagServe,
                                     tensorflow::kSavedModelTagTpu};
  tensorflow::SessionOptions session_options;
  session_options.target = tpu_target_;
  TF_CHECK_OK(tensorflow::LoadSavedModel(session_options, run_options,
                                         saved_model_path_, tags,
                                         &saved_model_bundle_));

  // Get names of input and output tensors from signature.
  auto iter = saved_model_bundle_.meta_graph_def.signature_def().find(
      tensorflow::kDefaultServingSignatureDefKey);
  if (iter == saved_model_bundle_.meta_graph_def.signature_def().end()) {
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("Could not find SignatureDef with key: serving_default"));
  }
  signature_def_ = iter->second;
  LOG(INFO) << "signature_def:" << signature_def_.DebugString();
  for (const auto& input : signature_def_.inputs()) {
    // TODO(chiachenc): supports arbitrary inputs.
    if (input.first == "indices") {
      input_tensor_names_ = input.second.name();
    }

    if (input.first == "image_list") {
      qsl_name_ = input.second.name();
    }
  }
  for (const auto& output : signature_def_.outputs()) {
    // TODO(chiachenc): supports arbitrary outputs.
    if (output.first == "logits") {
      output_tensor_names_.push_back(output.second.name());
    }

    if (output.first == "assign") {
      update_qsl_name_ = output.second.name();
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status Loader::CreateBatchingSession() {
  // Creates a batching session.
  tensorflow::serving::BatchingSessionOptions batching_session_options;
  for (int bs : batching_options_.batch_size) {
    batching_session_options.allowed_batch_sizes.push_back(bs);
  }

  tensorflow::serving::TensorSignature signature;
  signature.input_tensors.insert(input_tensor_names_);
  for (auto& name : output_tensor_names_) {
    signature.output_tensors.insert(name);
  }
  CHECK_LE(1, signature.input_tensors.size());
  tensorflow::serving::BasicBatchScheduler<
      tensorflow::serving::BatchingSessionTask>::Options schedule_options;
  schedule_options.thread_pool_name = batching_options_.thread_pool_name;
  schedule_options.num_batch_threads = batching_options_.num_batch_threads;
  schedule_options.batch_timeout_micros =
      batching_options_.batch_timeout_micros;
  schedule_options.max_enqueued_batches =
      batching_options_.max_enqueued_batches;
  schedule_options.max_batch_size = batching_options_.max_batch_size;

  TF_CHECK_OK(CreateBasicBatchingSession(
      schedule_options, batching_session_options, signature,
      std::move(saved_model_bundle_.session), &batching_session_));
  return tensorflow::Status::OK();
}

tensorflow::Status Loader::InitializeTpu() {
  LOG(INFO) << "Initializing TPU" << std::endl;
  tensorflow::GraphDef graph_def;
  tensorflow::protobuf::TextFormat::ParseFromString(kTpuOpsGraphDef,
                                                    &graph_def);

  tensorflow::SessionOptions options;
  options.target = tpu_target_;
  options.config.set_allow_soft_placement(true);
  options.config.set_log_device_placement(true);

  main_session_.reset(tensorflow::NewSession(options));
  TF_CHECK_OK(main_session_->Create(graph_def));
  TF_CHECK_OK(main_session_->Run({}, {}, {"ConfigureDistributedTPU"}, nullptr));
  return tensorflow::Status::OK();
}

tensorflow::Status Loader::ShutdownTpu() {
  // TODO(chiachenc): figure out how to shut down TPU.
  LOG(INFO) << "Shutting down TPU" << std::endl;
  return saved_model_bundle_.session->Run({}, {}, {"ShutdownDistributedTPU"},
                                          nullptr);
}

// TODO(tjablin): Add more tests.
void Loader::UpdateQSL(const tensorflow::Tensor& qsl) {
  std::vector<tensorflow::Tensor> outputs;
  TF_CHECK_OK(batching_session_->Run({{qsl_name_, qsl}}, {update_qsl_name_}, {},
                                     &outputs));
}

void Loader::Predict(const tensorflow::Tensor& inputs,
                     std::vector<tensorflow::Tensor>& outputs) {
  // TODO(chiachenc): supports arbitrary inputs.
  TF_CHECK_OK(batching_session_->Run({{input_tensor_names_, inputs}},
                                     output_tensor_names_, {}, &outputs));
}

}  // namespace mlperf
