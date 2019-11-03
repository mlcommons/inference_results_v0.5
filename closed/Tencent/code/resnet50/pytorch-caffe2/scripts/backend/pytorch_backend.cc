#include <string>
#include <vector>
#include "caffe2/core/flags.h"
#include "../classifier.h"
#include "../data_provider.h"
#include "../misc.h"
#include "inferencer.h"

C10_DEFINE_bool(dummy_data, false, "choose whether use dummy data or real data");
C10_DEFINE_bool(use_accuracy_layer, false, "use accuracy layers");
C10_DEFINE_bool(quantized, false, "if use quantized input");
C10_DEFINE_bool(random_multibatch, false, "use random multibatch input");
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

class BackendPytorch : public loadrun::Inferencer {
  public:
  BackendPytorch() {
    std:: cout << "Inferencer is opened\n";
  }

  ~BackendPytorch() {
    data_provider_->clean_shared_memory(FLAGS_numa_id);
    std:: cout << "Inferencer cleaned shared mem\n";
    delete classifier_;
    delete data_provider_;
    std:: cout << "Inferencer is closed\n";
  }

  void initialize(int argc, char **argv, bool use_index) {
    use_index_ = use_index;
    caffe2::GlobalInit(&argc, &argv);
    FLAGS_log_level == -1 ? FLAGS_caffe2_log_level = -1 :
                            FLAGS_caffe2_log_level = 0;
    // data provider can read from image path and handle data transformation process
    // like preprocess and shared memory control.
    data_provider_ = new caffe2::DataProvider<char>(FLAGS_file_list, FLAGS_images, FLAGS_labels,
                                          FLAGS_batch_size, FLAGS_data_order, FLAGS_dummy_data,
                                          FLAGS_iterations, FLAGS_net_conf,
                                          FLAGS_shared_memory_option, FLAGS_numa_id, use_index);

    // should give available iterations to construct the classifer.
    classifier_ = new caffe2::Classifier<char>(FLAGS_device_type, data_provider_->get_data(),
                                         data_provider_->get_labels(), FLAGS_init_net_path,
                                         FLAGS_predict_net_path, FLAGS_batch_size,
                                         FLAGS_data_order, FLAGS_use_accuracy_layer, 0,
                                         data_provider_->get_iterations(), FLAGS_net_conf,
                                         FLAGS_quantized, FLAGS_log_level,
                                         FLAGS_shared_weight, FLAGS_numa_id,
                                         FLAGS_dummy_data);
    classifier_->warmup(FLAGS_w);
    std::cout << "warmup done\n";
  }

  void prepare_batch(const int index){
    classifier_->random_input_prepare(index);
  }

  void run(int iteration, bool random){
    classifier_->run(iteration, random);
  }

  void accuracy(int iteration, bool random){
    classifier_->accuracy(iteration, random);
  }

  std::vector<int> get_labels(int iteration, bool random) {
    return classifier_->get_labels(iteration, random);
  }

  void load_sample(size_t* samples, size_t sample_size) {
    if (use_index_)
      data_provider_->load_sample(samples, FLAGS_batch_size * FLAGS_iterations);
  }

  void getInfo(double* hd_seconds, float* top1, float* top5) {
    classifier_->getInfo(hd_seconds, top1, top5);
  }

  void getInfo(double* run_seconds, double* hd_seconds, float* top1, float* top5) {
    classifier_->getInfo(run_seconds, hd_seconds, top1, top5);
  }

 private:
   caffe2::Classifier<char> *classifier_ = nullptr;
   caffe2::DataProvider<char> *data_provider_ = nullptr;
   bool use_index_ = false;
};

std::unique_ptr<loadrun::Inferencer> get_inferencer() {
  return std::unique_ptr<BackendPytorch>(new BackendPytorch());
}

