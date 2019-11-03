#include "caffe2/core/flags.h"
#include "classifier.h"
#include "data_provider.h"
#include "misc.h"

C10_DEFINE_bool(dummy_data, false, "choose whether use dummy data or real data");
C10_DEFINE_bool(use_accuracy_layer, false, "use accuracy layers");
C10_DEFINE_bool(quantized, false, "if use quantized input");
C10_DEFINE_bool(random_multibatch, false, "use random multibatch input");
C10_DEFINE_bool(loadgen_index, false, "use loadrun index");
C10_DEFINE_int(batch_size, 1, "image batch sizes");
C10_DEFINE_int(iterations, 1, "iterations");
C10_DEFINE_int(w, 5, "warm up times");
C10_DEFINE_int(log_level, 0, "log level: 0 means normal, -1 means profiling, -2 means iteration time");
C10_DEFINE_string(data_order, "NCHW", "data order for input data ");
C10_DEFINE_string(net_conf, "", "net configuration using yaml file");
C10_DEFINE_string(device_type, "ideep", "set device type, default ideep, you can aslo set cpu");
C10_DEFINE_string(images, "", "images path");
C10_DEFINE_string(labels, "", "labels files");
C10_DEFINE_string(file_list, "", "validation data path");
C10_DEFINE_string(init_net_path, "", "init model path");
C10_DEFINE_string(predict_net_path, "", "predict model path");
C10_DEFINE_string(numa_id, "0", "when shared memory used, set NUMA id to reduce"
                                "cross NUMA memory sharing");
C10_DEFINE_string(shared_memory_option, "USE_SHM", "choose shared memory options, options have:"
                                        "USE_SHM: directly use shared memory and create if not exists"
                                        "FREE_USE_SHM: free shared memory after using"
                                        "CREATE_USE_SHM: free shared memory and create a new one for using"
                                        "USE_LOCAL: use local memory and not care about shared memory");

C10_DEFINE_string(shared_weight, "USE_SHM", "choose shared weight options, options have:"
                                        "USE_SHM: directly use shared weight and create if not exists"
                                        "FREE_USE_SHM: free shared weight after using"
                                        "CREATE_USE_SHM: free shared weight and create a new one for using"
                                        "USE_LOCAL: use local weight and not care about shared weight");

template<typename T>
void RandomMultibatchTest(caffe2::Classifier<T>& classifier, caffe2::DataProvider<T>& data_provider) {
  LOG(INFO) << "random multibatch test begin";
  for (auto i = 0; i < FLAGS_iterations; ++i) {
    std::vector<int> indexes = [&FLAGS_batch_size, &i](){
      std::vector<int> v(FLAGS_batch_size, 0);
      // constuct a random indexes with index reversed
      for (int j = 0; j < FLAGS_batch_size; ++j) v[j] = j + i * FLAGS_batch_size;
      return v;
    }();
    for (size_t j = 0; j < indexes.size(); ++j) {
      classifier.random_input_prepare(indexes[j]);
      if (j == indexes.size() -1) {
        if (indexes.size() == 1) {
          classifier.run(0, true);
        } else classifier.run(0, true);
        // if (!FLAGS_dummy_data) classifier.accuracy(0, true);
        if (!FLAGS_dummy_data) classifier.get_labels();
      }
    }
  }
  // if (FLAGS_shared_memory_option == "FREE_USE_SHM") {
  //   LOG(INFO) << "shared memory will be free now";
  //   data_provider.clean_shared_memory(FLAGS_numa_id);
  // }
}

template<typename T>
void NormalMultibatchTest(caffe2::Classifier<T>& classifier, caffe2::DataProvider<T>& data_provider) {
  LOG(INFO) << "normal multibatch test begin";
  for (auto i = 0; i < FLAGS_iterations; i++) {
    classifier.random_input_prepare(i);
    classifier.run(i, false);
    if (!FLAGS_dummy_data) classifier.accuracy(i, false);
  }
  // if (FLAGS_shared_memory_option == "FREE_USE_SHM") {
  //   LOG(INFO) << "shared memory will be free now";
  //   data_provider.clean_shared_memory(FLAGS_numa_id);
  // }
}
template<typename T>
void run_net() {
  double e2e_seconds = 0;
  double run_seconds = 0;
  double hd_seconds = 0;
  float top1 = 0;
  float top5 = 0;
  caffe2::Timer timer;
  timer.Start();

  std::vector<std::size_t> samples;
  samples.resize(FLAGS_batch_size * FLAGS_iterations, 0);
  // std::string qsl = "qsl.txt";
  // std::ifstream qsl_stream(qsl); 
  for (std::size_t i = 0; i < FLAGS_batch_size * FLAGS_iterations; ++i) {
    // int temp_sample;
    // qsl_stream >> temp_sample;
    // samples[i] = static_cast<std::size_t>(temp_sample);
    // LOG(INFO) << "sample is "  << samples[i];
    samples[i] = i;
  }
  // qsl_stream.close();
 
  // data provider can read from image path and handle data transformation process
  // like preprocess and shared memory control.
  caffe2::DataProvider<T> data_provider(FLAGS_file_list, FLAGS_images, FLAGS_labels,
                                            FLAGS_batch_size, FLAGS_data_order, FLAGS_dummy_data,
                                            FLAGS_iterations, FLAGS_net_conf,
                                            FLAGS_shared_memory_option, FLAGS_numa_id, FLAGS_loadgen_index);

  // should give available iterations to construct the classifer.
  caffe2::Classifier<T> classifier(FLAGS_device_type, data_provider.get_data(),
                                       data_provider.get_labels(), FLAGS_init_net_path,
                                       FLAGS_predict_net_path, FLAGS_batch_size,
                                       FLAGS_data_order, FLAGS_use_accuracy_layer, 0,
                                       data_provider.get_iterations(), FLAGS_net_conf,
                                       FLAGS_quantized, FLAGS_log_level,
                                       FLAGS_shared_weight, FLAGS_numa_id,
                                       FLAGS_dummy_data);
  classifier.warmup(FLAGS_w);
  // iteration index may large than iterations available, same images will be processed.

  data_provider.load_sample(samples.data(), FLAGS_batch_size * FLAGS_iterations);

  if(FLAGS_random_multibatch) RandomMultibatchTest(classifier, data_provider);
  else NormalMultibatchTest(classifier, data_provider);
  classifier.getInfo(&run_seconds, &hd_seconds, &top1, &top5);
  e2e_seconds = timer.Seconds();
  double hd_perf = (FLAGS_batch_size * FLAGS_iterations) / hd_seconds;
  double e2e_perf = (FLAGS_batch_size * FLAGS_iterations) / e2e_seconds;
  double with_prepare_hd_perf = (FLAGS_batch_size * FLAGS_iterations) / run_seconds;
  LOG(INFO) << "e2e time is " << e2e_seconds; 
  LOG(INFO) << " hardware time is " << hd_seconds;
  LOG(INFO) << " run time is " << run_seconds;
  LOG(INFO) << "inference " << FLAGS_batch_size * FLAGS_iterations << " imgs takes: \n"
            << "end to end fps is " << e2e_perf << " images/second\n"
            << "with prepare fps is " << with_prepare_hd_perf << " images/second\n"
            << "hardware fps is " << hd_perf << " images/second";
  LOG(INFO) << "top1 accuracy is " << top1 / (FLAGS_batch_size * FLAGS_iterations);
  LOG(INFO) << "top5 accuracy is " << top5 / (FLAGS_batch_size * FLAGS_iterations);
}

int main(int argc, char **argv) {
  caffe2::GlobalInit(&argc, &argv);
  FLAGS_log_level == -1 ? FLAGS_caffe2_log_level = -1 :
                          FLAGS_caffe2_log_level = 0;

  if (FLAGS_quantized) run_net<char>();
  else run_net<float>();
  return 0;
}
