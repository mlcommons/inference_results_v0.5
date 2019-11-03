#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <stdlib.h>

#include <loadgen.h>
#include <query_sample_library.h>
#include <system_under_test.h>
#include <test_settings.h>

#include "dg.h"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

//#include "tensorflow/lite/delegates/gpu/gl_delegate.h"

#define NOLOWER
#include "lower_helper.h"

using namespace mlperf;
using namespace std;


TestSettings testSettings;
std::vector<std::vector<char>> input_images;
bool warmup = true;

class FuriosaSystemUnderTest : public SystemUnderTest {
private:
  const std::string name = "FuriosaAI";
  std::vector<QuerySampleResponse> res_;

  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;

public:

  FuriosaSystemUnderTest() {

#ifdef NOLOWER
    const char *model_filename = "models/mobilenet_without_lower.tflite";
#else
    const char *model_filename = "models/mobilenet_v1.tflite";
#endif
    model_ = tflite::FlatBufferModel::BuildFromFile(model_filename);
    if (model_ == nullptr) {
      std::cerr << "failed to load model.\n";
      throw std::runtime_error("failed to load model.");
    }

    cerr << "create interpreter\n";
    interpreter_ = make_furiosa_interpreter(*model_, 1);

  //tflite::ops::builtin::BuiltinOpResolver resolver;
  //tflite::InterpreterBuilder builder(*model_, resolver);
  //builder(&interpreter_);

    if (interpreter_ == nullptr) {
      std::cerr << "failed to build interpreter.\n";
      throw std::runtime_error("failed to build interpreter.");
    }

    cerr << "resize input tensor\n";
#ifdef NOLOWER
    auto ret = interpreter_->ResizeInputTensor(interpreter_->inputs()[0], std::vector<int>{224*224*4});
#else
    auto ret = interpreter_->ResizeInputTensor(interpreter_->inputs()[0], std::vector<int>{1,224,224,3});
#endif
    std::cerr <<"RESIZE " << (int) ret << "\n";

    ret = interpreter_->AllocateTensors();
  }

  ~FuriosaSystemUnderTest() = default;

  const std::string &Name() const { return name; }

  void IssueQuery(const std::vector<QuerySample> &samples) {
    for (QuerySample sample : samples) {
      auto& data = input_images[sample.index];
      // std::cerr << "SAMPLE " << (int)sample.index << ' ' << data.size() << "\n";
      // std::cerr << interpreter_->inputs().size() << ' ' << interpreter_->inputs()[0] << "\n";
      for (size_t i = 0; i < interpreter_->inputs().size(); i++) {
        auto tensor = interpreter_->tensor(interpreter_->inputs()[i]);
        auto buf = tensor->data.raw;
        // std::cerr << "copy " << tensor->bytes << ' ' << (void*)buf << ' ' << data.size() << '\n';
        memcpy(buf, &data[0], data.size());
      }
      // std::cerr << "Invoke\n";
      interpreter_->Invoke();

      // std::cerr << "Invoke done\n";
      auto t = interpreter_->tensor(interpreter_->outputs()[0]);
      // std::cerr << "t "<<t<<"\n";

      int max_idx = 0;
      uint8_t max_data = t->data.uint8[0];
      for(int i = 1; i < (int) t->bytes; i ++)
        if (max_data < t->data.uint8[i]) {
          max_idx = i-1;
          max_data = t->data.uint8[i];
        }

      QuerySampleResponse res;
      res.id = sample.id;
      res.data = (uintptr_t)&max_idx;
      res.size = 4;
      // std::cerr << "report " << max_idx << '\n';

      if( ! warmup )
        QuerySamplesComplete(&res, 1);
    }
  }

  void FlushQueries() {}

  void
  ReportLatencyResults(const std::vector<QuerySampleLatency> &/*latencies_ns*/) {}
};

class FuriosaQuerySampleLibrary : public QuerySampleLibrary {
private:
  const std::string name = "FuriosaAI";

  std::vector<std::pair<std::string, int>> items_;

public:
  FuriosaQuerySampleLibrary() {
    std::ifstream val_file((
        std::string("/home/furiosa/") + "/CK-TOOLS/dataset-imagenet-ilsvrc2012-val/val_map.txt").c_str()
      );

    std::string input_filename;
    int answer;
    while (val_file >> input_filename >> answer) {
      items_.emplace_back(input_filename.substr(0, input_filename.size()-5)+".bin", answer);
    }
    // items_.resize(10000);
    input_images.resize(items_.size());
  }

  ~FuriosaQuerySampleLibrary() {}

  const std::string &Name() const { return name; }

  size_t TotalSampleCount() { return items_.size(); }

  size_t PerformanceSampleCount() { return items_.size(); }

  void LoadSamplesToRam(const std::vector<QuerySampleIndex> &samples) override {

    for (auto index : samples) {
      std::string filename = "/preprocessed/" + items_[index].first;
      filename = "/home/furiosa" + filename;

      std::ifstream inf(filename.c_str(), std::ios::binary);

#ifdef NOLOWER
      std::vector<char> temp, temp1;
      temp.resize(224*224*3);
      inf.read((char*)&temp[0], 224*224*3);
      temp1.resize(224*224*3);
      transpose(&temp[0], &temp1[0], 16, 14, 224, 3);
      input_images[index].resize(224*224*4);
      pad(&temp1[0], &input_images[index][0], 16, 3*224*14, 4*224*14);

#else
      input_images[index].resize(224 * 224 * 3);
      inf.read((char*)&input_images[index][0], 224*224*3);
#endif
    }
  }

  void UnloadSamplesFromRam(const std::vector<QuerySampleIndex> &samples) override {
    for (auto index : samples) {
      std::vector<char>().swap(input_images[index]);
      input_images[index].clear();
      input_images[index].shrink_to_fit();
    }
  }
};

int main(int argc, char **argv) {

  unsigned int minq = 1000; // should be > 25K for submission.

  if( argc == 2 ) // query count;
    minq = stoi(argv[1], nullptr, 10);

  cerr << "create sut\n";
  SystemUnderTest *sut = new FuriosaSystemUnderTest();

  cerr << "create qsl\n";
  QuerySampleLibrary *qsl = new FuriosaQuerySampleLibrary();

  testSettings = TestSettings();
  testSettings.scenario = TestScenario::SingleStream;
  // testSettings.mode = TestMode::SubmissionRun;
  testSettings.mode = TestMode::PerformanceOnly;
  // testSettings.single_stream_expected_latency_ns = 1000000;
  testSettings.min_query_count = minq;
  testSettings.max_query_count = minq;
  testSettings.min_duration_ms = 60000;

  testSettings.qsl_rng_seed = 0x2b7e151628aed2a6ULL;
  testSettings.schedule_rng_seed = 0x3243f6a8885a308dULL;
  testSettings.sample_index_rng_seed = 0x093c467e37db0c7aULL;

  LogSettings logSettings = LogSettings();

  logSettings.enable_trace = false;

  cerr << "mobilenet v1 --- Single Stream\n";
  cerr << "Query count: " << minq << "\n";

  warmup = false;
  cerr << "Warmed up!\n";

  cerr << "Start loadgen\n";
  StartTest(sut, qsl, testSettings, logSettings);

  return 0;
}
