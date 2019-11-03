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

#include <tbb/concurrent_queue.h>
#include <tbb/parallel_for.h>


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

using namespace mlperf;
using nlohmann::json;
using namespace std;

int N = 4;
auto start_time = std::chrono::high_resolution_clock::now();


std::vector<std::vector<char>> input_images;

std::string get_base_dir() {
  return "/home/furiosa";
}

class FuriosaSystemUnderTest : public SystemUnderTest {
private:
  const std::string name = "FuriosaAI";
  std::vector<QuerySampleResponse> res_;

  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::vector<std::unique_ptr<tflite::Interpreter>> interpreters_;

  std::vector<std::thread> pool;
  tbb::concurrent_queue<QuerySample> works;
  std::atomic<bool> done{false};

  int qcount = 0;
  int scount = 0;

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

    interpreters_ = make_furiosa_interpreters(*model_, N, N, N);
    for(int i = 0; i < interpreters_.size(); i ++) {
      auto interpreter_ = interpreters_[i].get();

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

      if (ret != kTfLiteOk) {
        std::cerr << "failed to resize input tensor.\n";
        throw std::runtime_error("failed to resize input tensor.");
      }

      ret = interpreter_->AllocateTensors();

      if (ret != kTfLiteOk) {
        std::cerr << "failed to allocate tensors.\n";
        throw std::runtime_error("failed to allocate tensors.");
      }

      // create threads
      pool.emplace_back(std::thread([&, idx = i] {
				  // std::cerr << "worker " << idx << " start!\n";
			    auto interpreter_ = interpreters_[idx].get();
				  while(!done || !works.empty()) {
				    QuerySample sample;
				    if (!works.try_pop(sample)) {
					    std::this_thread::yield();
					    continue;
				    }

            // std::cerr << "Works remain: " <<works.unsafe_size() << "\n";

            auto& data = input_images[sample.index];
            //std::cerr << "SAMPLE " << (int)sample.index << ' ' << data.size() << "\n";
            //std::cerr << interpreter_->inputs().size() << ' ' << interpreter_->inputs()[0] << "\n";
            for (size_t i = 0; i < interpreter_->inputs().size(); i++) {
              auto tensor = interpreter_->tensor(interpreter_->inputs()[i]);
              auto buf = tensor->data.raw;
              memcpy(buf, &data[0], data.size());
            }
            //std::cerr << "Invoke\n";
            interpreter_->Invoke();

            //std::cerr << "Invoke done\n";


            auto t1 = interpreter_->tensor(interpreter_->outputs()[0]);
            auto t2 = interpreter_->tensor(interpreter_->outputs()[1]);
            auto t3 = interpreter_->tensor(interpreter_->outputs()[2]);
            auto t0 = interpreter_->tensor(interpreter_->outputs()[3]);
            int n_det = (int)(t0->data.f[0]);
            thread_local std::vector<float> buffer;
            buffer.resize(n_det*7);

            float* out_p = &buffer[0];
            for(int i = 0; i < n_det; i ++) {
                //image_idx, ymin, xmin, ymax, xmax, score, label = data[i:i + 7]
              *out_p++ = (float)sample.index;
              *out_p++ = t1->data.f[i*4+0];
              *out_p++ = t1->data.f[i*4+1];
              *out_p++ = t1->data.f[i*4+2];
              *out_p++ = t1->data.f[i*4+3];
              *out_p++ = t3->data.f[i];
              *out_p++ = t2->data.f[i]+1;
            }

            QuerySampleResponse res;
            res.id = sample.id;
            res.data = (uintptr_t)&buffer[0];
            res.size = buffer.size()*sizeof(float);
            // std::cerr << "report " << max_idx << '\n';

            QuerySamplesComplete(&res, 1);
				  }
			  }
      ));

    }
  }

  ~FuriosaSystemUnderTest() = default;

  const std::string &Name() const { return name; }

  void IssueQuery(const std::vector<QuerySample> &samples) {
    int query_size = samples.size();
    tbb::parallel_for(0, query_size, [&] (int i)
    {
        works.push(samples[i]);
    });
  }

  void FlushQueries() {
    // std::cerr << "start wait\n";

    done = true;

    while(!works.empty())
    {
      std::this_thread::yield();
    }

    for(auto& t:pool)
	    t.join();
  }

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
  int spq = 24576;
  // unsigned int minq = 1000; // should be > 280K for submission.

  if( argc > 1 )
    N = stoi(argv[1], nullptr, 10);

  if( argc > 2 ) // samples per query;
    spq = stoi(argv[2], nullptr, 10);

  std::cerr << "create sut\n";
  SystemUnderTest *sut = new FuriosaSystemUnderTest();

  std::cerr << "create qsl\n";
  QuerySampleLibrary *qsl = new FuriosaQuerySampleLibrary();

  TestSettings testSettings = TestSettings();
  testSettings.scenario = TestScenario::Offline;
  testSettings.mode = TestMode::PerformanceOnly;

  testSettings.min_query_count = spq;
  testSettings.max_query_count = spq;
  testSettings.min_duration_ms = 60000;

  testSettings.qsl_rng_seed = 0x2b7e151628aed2a6ULL;
  testSettings.schedule_rng_seed = 0x3243f6a8885a308dULL;
  testSettings.sample_index_rng_seed = 0x093c467e37db0c7aULL;

  LogSettings logSettings = LogSettings();

  logSettings.enable_trace = false;

  cerr << "ssd-mobilenet v1 --- Offline\n";

  std::cerr << "Samples per second: " << spq << "\n";
  std::cerr << "Query count: " << 1 << "\n";
  cerr << "# NPUs enabled: " << N << "\n";

  cerr << "Start loadgen\n";
  StartTest(sut, qsl, testSettings, logSettings);

  return 0;
}
