#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <chrono>
#include "json.hpp"

#include <loadgen.h>
#include <query_sample_library.h>
#include <system_under_test.h>
#include <test_settings.h>

#include "dg.h"

//#define STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_RESIZE_IMPLEMENTATION
//#include "stb_image.h"
//#include "stb_image_resize.h"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

//#include "tensorflow/lite/delegates/gpu/gl_delegate.h"

//#define NOLOWER
#include "lower_helper.h"

using namespace mlperf;
using nlohmann::json;
using namespace std;


std::vector<std::vector<char>> input_images;
bool warmup = true;

std::string get_base_dir() {
  return "/home/furiosa";
}

class FuriosaSystemUnderTest : public SystemUnderTest {
private:
  const std::string name = "FuriosaAI";
  std::vector<QuerySampleResponse> res_;

  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
public:
  FuriosaSystemUnderTest() {

#ifdef NOLOWER
    const char *model_filename = "models/ssd300_without_lower.tflite";
#else
    const char *model_filename = "models/ssd_mobilenet_v1.tflite";
#endif
    model_ = tflite::FlatBufferModel::BuildFromFile(model_filename);
    if (model_ == nullptr) {
      std::cerr << "failed to load model.\n";
      throw std::runtime_error("failed to load model.");
    }

    interpreter_ = make_furiosa_interpreter(*model_, 1, 1);
  //tflite::ops::builtin::BuiltinOpResolver resolver;
  //tflite::InterpreterBuilder builder(*model_, resolver);
  //builder(&interpreter_);

    if (interpreter_ == nullptr) {
      std::cerr << "failed to build interpreter.\n";
      throw std::runtime_error("failed to build interpreter.");
    }

#ifdef NOLOWER
    auto ret = interpreter_->ResizeInputTensor(interpreter_->inputs()[0], std::vector<int>{320*320*4});
#else
    auto ret = interpreter_->ResizeInputTensor(interpreter_->inputs()[0], std::vector<int>{1,300,300,3});
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
        memcpy(buf, &data[0], data.size());
      }
      // std::cerr << "Invoke\n";
      interpreter_->Invoke();

      // std::cerr << "Invoke done\n";


      auto t1 = interpreter_->tensor(interpreter_->outputs()[0]);
      // std::cerr << "output 1\n";
      auto t2 = interpreter_->tensor(interpreter_->outputs()[1]);
      // std::cerr << "output 2\n";
      auto t3 = interpreter_->tensor(interpreter_->outputs()[2]);
      // std::cerr << "output 3\n";
      auto t0 = interpreter_->tensor(interpreter_->outputs()[3]);

      // std::cerr << "Got tensor outputs\n";

      int n_det = (int)(t0->data.f[0]);
      thread_local std::vector<float> buffer;
      buffer.resize(n_det*7);

      // std::cerr << "Buffer resized\n";

      float* out_p = &buffer[0];

      // std::cerr << "before for loop\n";

      for(int i = 0; i < n_det; i ++) {
        // image_idx, ymin, xmin, ymax, xmax, score, label = data[i:i + 7]
        *out_p++ = (float)sample.index;
        *out_p++ = t1->data.f[i*4+0];
        *out_p++ = t1->data.f[i*4+1];
        *out_p++ = t1->data.f[i*4+2];
        *out_p++ = t1->data.f[i*4+3];
        *out_p++ = t3->data.f[i];
        *out_p++ = t2->data.f[i]+1;
      }

      // std::cerr << "after for loop\n";

      QuerySampleResponse res;
      res.id = sample.id;
      res.data = (uintptr_t)&buffer[0];
      res.size = buffer.size()*sizeof(float);

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

  std::vector<std::string> items_;

public:
  FuriosaQuerySampleLibrary() {
    std::ifstream val_file((
        get_base_dir() + "/CK-TOOLS/dataset-coco-2017-val/annotations/instances_val2017.json").c_str());

    json j;
    val_file >> j;
    items_.reserve(5000);
    for(auto& item:j["images"]) {
      std::string fname = item["file_name"].get<std::string>();

      items_.emplace_back(get_base_dir() + "/coco_preprocessed/" + fname.substr(0,12) + ".bin");
    }
    input_images.resize(items_.size());
    std::cerr << "LOADED " << items_.size() << " inputs. (" << items_[0] << ")\n";
  }

  ~FuriosaQuerySampleLibrary() {}

  const std::string &Name() const { return name; }

  size_t TotalSampleCount() { return items_.size(); }

  size_t PerformanceSampleCount() { return 5000; }

  void LoadSamplesToRam(const std::vector<QuerySampleIndex> &samples) override {

    for (auto index : samples) {
      //std::string filename = "~/CK-TOOLS/dataset-imagenet-ilsvrc2012-val/" + items_[index];
      //filename = getenv("HOME") + filename;
      std::string filename = items_[index];

      std::ifstream inf(filename.c_str(), std::ios::binary);

#ifdef NOLOWER
      std::vector<char> temp, temp1;
      temp.resize(300*300*3);
      temp1.resize(320*320*4);
      char*a = &temp[0];
      char*b = &temp1[0];
    for(int y = 0; y < 300; y ++) {
      for(int i = 0; i < 300;i++) {
        for(int j = 0; j < 3; j ++)
          *b++ = *a++;
        for(int j = 0; j < 1;j++)
          *b++ = 128;
      }
      for(int i = 0; i < 20; i ++) {
        for(int j = 0; j < 4; j ++)
	  *b++ = 128;
      }
    }
    for(int y = 0; y < 20; y ++) {
      for(int x = 0; x < 320;x++) {
        for(int j = 0; j < 4;j++)
          *b++ = 128;
      }
    }
    // 300x300x4
    input_images[index].resize(320*320*4);
    transpose(&temp1[0], &input_images[index][0], 16, 20, 320, 4);
#else
      input_images[index].resize(300 * 300 * 3);
      inf.read((char*)&input_images[index][0], 300*300*3);
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

  SystemUnderTest *sut = new FuriosaSystemUnderTest();

  QuerySampleLibrary *qsl = new FuriosaQuerySampleLibrary();

  TestSettings testSettings = TestSettings();
  testSettings.scenario = TestScenario::SingleStream;
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


  cerr << "ssd-mobilenet v1 --- Single Stream\n";
  cerr << "Query count: " << minq << "\n";
  warmup = false;
  cerr << "Warmed up!\n";


  cerr << "Start loadgen\n";
  StartTest(sut, qsl, testSettings, logSettings);

  return 0;
}
