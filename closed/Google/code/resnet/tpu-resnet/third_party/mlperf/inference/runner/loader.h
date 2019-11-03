#ifndef MLPERF_INFERENCE_RUNNER_LOADER_H_
#define MLPERF_INFERENCE_RUNNER_LOADER_H_

#include <string>
#include <vector>

#include "mlperf/inference/runner/options.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow_serving/batching/batching_session.h"

namespace mlperf {

class Loader {
 public:
  Loader(const std::string tpu_target, const std::string saved_model_path,
         const mlperf::BatchingOption batching_options, bool init_tpu);

  void UpdateQSL(const tensorflow::Tensor& qsl);

  void Predict(const tensorflow::Tensor& inputs,
               std::vector<tensorflow::Tensor>& outputs);

 private:
  tensorflow::Status LoadSavedModel();
  tensorflow::Status CreateBatchingSession();
  tensorflow::Status InitializeTpu();
  tensorflow::Status ShutdownTpu();

  std::string tpu_target_;
  std::string saved_model_path_;
  mlperf::BatchingOption batching_options_;

  tensorflow::SavedModelBundle saved_model_bundle_;
  tensorflow::SignatureDef signature_def_;

  std::string qsl_name_;
  std::string update_qsl_name_;
  std::string input_tensor_names_;
  std::vector<std::string> output_tensor_names_;

  std::unique_ptr<tensorflow::Session> batching_session_;
  std::unique_ptr<tensorflow::Session> main_session_;
};
}  // namespace mlperf

#endif  // MLPERF_INFERENCE_RUNNER_LOADER_H_
