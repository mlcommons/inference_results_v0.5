#ifndef CLASSIFIER_H__
#define CLASSIFIER_H__
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <sstream>
#include <string>
#include <vector>
#include "caffe2/core/flags.h"
#include "caffe2/core/init.h"
#include "caffe2/core/blob.h"
#include "caffe2/core/timer.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/ideep/ideep_utils.h"
#include "caffe2/observers/time_observer.h"
#include "caffe2/opt/optimize_ideep.h"
#include "caffe2/opt/converter.h"
#include "net_config.h"
#include "misc.h"

namespace caffe2 {

using namespace boost::interprocess;

template<typename T>
class Classifier {
 public:
    Classifier(const string& device_type, T* input_data, const vector<int>& labels,
               const string& init_net_path, const string& predict_net_path,
               const int batch_size, const string& data_order, const bool use_accuracy_layer,
               const int thread_id, const int iterations,
               const string& net_conf, const bool quantized, const int log_level,
               const string& shared_memory_option, const string& numa_id,
               const bool dummy_data);
    ~Classifier();
    void warmup(int warmup_times);
    int random_input_prepare(const int index);
    int run(int iteration, const bool random = true);
    void run(const vector<int>& iterations, const bool random = false);
    void accuracy(int iteration = 0, const bool random = false);
    void getInfo(double* hd_seconds, float* top1, float* top5);
    void getInfo(double* run_seconds, double* hd_seconds, float* top1, float* top5);
    vector<int> get_labels(int iteration = 0, const bool random = true);
  private:
    std::unique_ptr<NetConf> net_conf_;
    string device_type_;
    vector<int> labelsOut_;
    const Observable<NetBase>::Observer* observer_ = nullptr;
    T* inputData_;
    managed_shared_memory managed_shm_;
    const vector<int>& labels_;
    const bool quantized_ = false;
    const bool accuracy_ = false;
    const bool dummy_data_ = false;
    const int log_level_ = 0;
    int thread_id_ = 0;
    int thread_step_ = 0;
    int batchSize_ = 1;
    int iterations_ = 1;
    int random_count_ = 0;
    unsigned long long  inputSize_ = 0;
    float top1_ = 0;
    float top5_ = 0;
    double hd_seconds_ = 0;
    double runtime_seconds_ = 0;
    string dataOrder_ = "NCHW";
    string sharedMemory_; 
    string numaId_;
    caffe2::NetDef init_net_, predict_net_;
    caffe2::NetDef new_predict_net_;
    DeviceOption op_;
    Workspace ws_;
    vector<int> input_shape_;
    vector<string> output_blob_names_;
    vector<string> input_blob_names_;
    vector<int> randomLabels_;
    vector<T> randomInputImgs_;
    void AccuracyCompute(const int* labels);
    void InitNet(const string& device_type,
                 const string& init_net_path, const string& predict_net_path);
    void RandomPrepareInputBlob(const vector<int>& indexes);
    void PrepareInputBlob(T* inputImgs, const int* labels);
    void PrepareWeightBlob();
    void* GetBlobData(const string& blob_name, string& data_type, int* N, int* D);
    template<typename M>
    void AssignSharedWeight(M* blob_data, M* shared_weight,
                            const string& blob_name, size_t tensor_size);
    template<typename M>
    void SetDataHandle(M* shared_weight, const string& blob_name);
    // methods for shared weights
    void CreateUseSharedWeight();
    void DirectUseSharedWeight();
    void CleanSharedWeight(const string& numa_id);
    // optimazation batchsize = 1 situation
    int one_idx_ = 0;
};

void SetDeviceType(caffe2::NetDef *net_def, google::protobuf::int32 run_dev) {
  for (int j = 0; j < net_def->op_size(); j++) {
    caffe2::OperatorDef *op = net_def->mutable_op(j);
    op->mutable_device_option()->set_device_type(run_dev);
  }
}

template<typename T>
Classifier<T>::Classifier(const string& device_type, T* input_data, const vector<int>& labels,
               const string& init_net_path, const string& predict_net_path,
               const int batch_size, const string& data_order, const bool use_accuracy_layer,
               const int thread_id, const int iterations,
               const string& net_conf, const bool quantized, const int log_level,
               const string& shared_memory_option, const string& numa_id,
               const bool dummy_data)
  : device_type_(device_type),
    inputData_(input_data),
    labels_(labels),
    quantized_(quantized),
    accuracy_(use_accuracy_layer),
    dummy_data_(dummy_data),
    log_level_(log_level),
    thread_id_(thread_id),
    batchSize_(batch_size),
    iterations_(iterations),
    dataOrder_(data_order),
    sharedMemory_(shared_memory_option),
    numaId_(numa_id){
  net_conf_ = get_net_conf(net_conf);
  LOG(INFO) << "net name is " << net_conf_->net_name;
  inputSize_ = net_conf_->channels * net_conf_->height * net_conf_->width;
  randomInputImgs_.resize(batchSize_ * inputSize_, 0);
  randomLabels_.resize(batchSize_, -1);
  labelsOut_.resize(batchSize_, -1);

  InitNet(device_type, init_net_path, predict_net_path);
}
template<typename T>
Classifier<T>::~Classifier() {
  // remove the observer from net.
  if (log_level_ == -1) {
    auto net = ws_.GetNet(new_predict_net_.name());
    net->DetachObserver(observer_);
  }
}

template<typename T>
void Classifier<T>::InitNet(const string& device_type,
                         const string& init_net_path, const string& predict_net_path) {
  if (init_net_path.empty() || predict_net_path.empty()) {
    LOG(FATAL) << "init and predict net path should be given!";
  } else {
    ReadProtoFromFile(init_net_path, &init_net_);
    ReadProtoFromFile(predict_net_path, &predict_net_);
  }

  if (device_type == "ideep") {
    op_.set_device_type(PROTO_IDEEP);
    SetDeviceType(&init_net_, PROTO_IDEEP);
    SetDeviceType(&predict_net_, PROTO_IDEEP);
  } else if (device_type == "cpu") {
    op_.set_device_type(PROTO_CPU);
    SetDeviceType(&init_net_, PROTO_CPU);
    SetDeviceType(&predict_net_, PROTO_CPU);
  } else LOG(FATAL) << "unknown device type!";

  LOG(INFO) << "mean value is " << net_conf_->mean_value[0] <<
                           " : " << net_conf_->mean_value[1] <<
                           " : " << net_conf_->mean_value[2];
  input_shape_ = {batchSize_, net_conf_->channels, net_conf_->height, net_conf_->width};
  // input should be placed in external position 0 in the pb file.
  input_blob_names_.push_back(predict_net_.external_input(0));
  if (accuracy_)
    input_blob_names_.push_back(predict_net_.external_input(1));
  // input image has relation with thread id and iterations per thread
  // so every classifier will deal with images from thread_id_ * iterations_
  for (auto input_blob : input_blob_names_)
    ws_.CreateBlob(input_blob);

  PrepareInputBlob(inputData_, randomLabels_.data());
  CAFFE_ENFORCE(ws_.RunNetOnce(init_net_));
  auto nn = caffe2::convertToNNModule(predict_net_);
  if (device_type == "ideep")
    opt::OptimizeForMkldnn(&nn, &ws_, false);
  new_predict_net_ = caffe2::convertToCaffe2Proto(nn, predict_net_);
  auto net = ws_.CreateNet(new_predict_net_);
  CAFFE_ENFORCE(net);

  // for operator profiling use, create time observer for net.
  if (log_level_ == -1) {
    std::unique_ptr<TimeObserver> net_ob = std::unique_ptr<TimeObserver>(new TimeObserver(net));
    observer_ = net->AttachObserver(std::move(net_ob));
    CAFFE_ENFORCE(observer_);
  }
  output_blob_names_ = net->external_output();
  PrepareWeightBlob();
}
template<typename T>
void Classifier<T>::warmup(int warmup_times) {
  LOG(INFO) << "Warmup ..." << warmup_times;
  for (auto j = 0; j < warmup_times; j++) {
      ws_.RunNet(new_predict_net_.name());
  }

}

template<typename T>
void Classifier<T>::PrepareWeightBlob() {
  if (sharedMemory_ == "CREATE_USE_SHM") {
    CleanSharedWeight(numaId_);
    CreateUseSharedWeight();
  } else if (sharedMemory_ == "USE_SHM"){
    DirectUseSharedWeight();
  }
}

template<typename T>
void Classifier<T>::CleanSharedWeight(const string& numa_id) {
  shared_memory_object::remove(("SharedWeight" + numa_id).c_str());
}

template<typename T>
void* Classifier<T>::GetBlobData(const string& blob_name,
                                string& data_type, int* N, int* D) {
  void* raw_data = nullptr;
  auto data_blob = ws_.GetBlob(blob_name);
  Tensor cpuInputTensor;
  ideep::tensor ideepInputTensor; 
  if (BlobIsTensorType(*data_blob, CPU)) {
    cpuInputTensor = data_blob->Get<Tensor>().Clone();
    *N = cpuInputTensor.dim32(0);
    *D = cpuInputTensor.numel() / *N;
    auto meta = cpuInputTensor.dtype();
    std::stringstream meta_type;
    meta_type << meta;
    meta_type >> data_type;
    if (data_type == "long") {
      raw_data = static_cast<long*>(cpuInputTensor.raw_data());
    } else if (data_type == "float"){
      raw_data = static_cast<float*>(cpuInputTensor.raw_data());
    }
  } else {
    ideepInputTensor = data_blob->Get<ideep::tensor>();
    *N = static_cast<int>(ideepInputTensor.get_dim(0));
    *D = static_cast<int>(ideepInputTensor.get_nelems() / *N);
    if (ideepInputTensor.get_data_type() == ideep::tensor::data_type::s8) {
      data_type = "char";
      raw_data = static_cast<char*>(ideepInputTensor.get_data_handle());
    } else {
      data_type = "float";
      raw_data = static_cast<float*>(ideepInputTensor.get_data_handle());
    }
  }
  return raw_data;
}

template<typename T>
template<typename M>
void Classifier<T>::AssignSharedWeight(M* blob_data, M* shared_weight,
                                       const string& blob_name, size_t tensor_size) {
  const allocator<M, managed_shared_memory::segment_manager>
    alloc_inst(managed_shm_.get_segment_manager());
  auto shared_memory =
    managed_shm_.find_or_construct<vector<M, allocator<M,
      managed_shared_memory::segment_manager>>>((blob_name + numaId_).c_str())(alloc_inst);

  shared_memory->resize(tensor_size);
  std::memcpy(shared_memory->data(), blob_data, tensor_size * sizeof(M));
  shared_weight = shared_memory->data();
}

template<typename T>
template<typename M>
void Classifier<T>::SetDataHandle(M* shared_weight, const string& blob_name) {
  auto data_blob = ws_.GetBlob(blob_name);
  Tensor cpuInputTensor;
  ideep::tensor ideepInputTensor; 
  if (BlobIsTensorType(*data_blob, CPU)) {
    cpuInputTensor = data_blob->Get<Tensor>().Clone();
    cpuInputTensor.ShareExternalPointer(shared_weight);
  } else {
    ideepInputTensor = data_blob->Get<ideep::tensor>();
    ideepInputTensor.set_data_handle(shared_weight);
  }
}

template<typename T>
void Classifier<T>::CreateUseSharedWeight() {
  const unsigned long long TOTAL_WEIGHT_SIZE = 1024 * 1024 * 8;
  std::vector<string> weight_blob_names(
    // means we should cut out input activation for weights
    predict_net_.external_input().begin() + (accuracy_ ? 2 : 1),
    predict_net_.external_input().end());
  managed_shm_ = managed_shared_memory(open_or_create,
    ("SharedWeight" + numaId_).c_str() , 1024 * TOTAL_WEIGHT_SIZE);
  // check whether shared memory has prepared target image data, if not, prepare target data.
  auto shared_weight_size =
       managed_shm_.find_or_construct<int>(("SharedWeightSize" + numaId_).c_str())(0);
 
  // weight now only support fp32 and you should not use template type char
  LOG(INFO) << "weight blob size is " << weight_blob_names.size();
  for (int i = 0; i < weight_blob_names.size(); ++i) {
    string data_type;
    auto data_blob = ws_.GetBlob(weight_blob_names[i]);
    float* blob_data;
    size_t N;
    Tensor cpuInputTensor;
    ideep::tensor ideepInputTensor; 
    if (BlobIsTensorType(*data_blob, CPU)) {
      cpuInputTensor = data_blob->Get<Tensor>().Clone();
      N = cpuInputTensor.numel();
      auto meta = cpuInputTensor.dtype();
      std::stringstream meta_type;
      meta_type << meta;
      string data_type;
      meta_type >> data_type;
      blob_data = static_cast<float*>(cpuInputTensor.raw_data());
    } else {
      ideepInputTensor = data_blob->Get<ideep::tensor>();
      N = ideepInputTensor.get_nelems();
      if (ideepInputTensor.get_data_type() == ideep::tensor::data_type::s8) {
        auto s8_blob_data = static_cast<char*>(ideepInputTensor.get_data_handle());
        const allocator<char, managed_shared_memory::segment_manager>
          alloc_inst(managed_shm_.get_segment_manager());
        auto shared_weight =
          managed_shm_.find_or_construct<vector<char, allocator<char,
            managed_shared_memory::segment_manager>>>((weight_blob_names[i] + numaId_).c_str())(alloc_inst);
        shared_weight->resize(N);
        std::memcpy(shared_weight->data(), s8_blob_data, ideepInputTensor.get_nelems() * sizeof(char));
        ideepInputTensor.set_data_handle(shared_weight->data());
        (*shared_weight_size)++;
        continue;
      } else {
        blob_data = static_cast<float*>(ideepInputTensor.get_data_handle());
      }
    }

    const allocator<float, managed_shared_memory::segment_manager>
      alloc_inst(managed_shm_.get_segment_manager());
    auto shared_weight =
      managed_shm_.find_or_construct<vector<float, allocator<T,
        managed_shared_memory::segment_manager>>>((weight_blob_names[i] + numaId_).c_str())(alloc_inst);

    shared_weight->resize(N);
    std::memcpy(shared_weight->data(), blob_data, N * sizeof(float));
    if (BlobIsTensorType(*data_blob, CPU)) {
      cpuInputTensor.ShareExternalPointer(shared_weight->data());
    } else {
      ideepInputTensor.set_data_handle(shared_weight->data());
    }
    (*shared_weight_size)++;
  }
}

template<typename T>
void Classifier<T>::DirectUseSharedWeight() {
  std::vector<string> weight_blob_names(
    // means we should cut out input activation for weights
    predict_net_.external_input().begin() + (accuracy_ ? 2 : 1),
    predict_net_.external_input().end());
  int temp_status = 0;
  while (temp_status == 0) {
    try {
      managed_shm_ = managed_shared_memory(open_only, ("SharedWeight" + numaId_).c_str());
      temp_status = 1;
    }
    catch(...) {
      LOG(INFO) << "check whether shared weight created, use CREATE_USE_SHM in command line"; 
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  temp_status = 0;
  // check whether images has been preprocessed
  while (*(managed_shm_.find<int>(("SharedWeightSize" + numaId_).c_str()).first)
         != weight_blob_names.size()) {
    if (temp_status == 0) {
      LOG(INFO) << "shared weight size not satisfied, wait swap weights completed"; 
      //set temp_status to 1 because we only want log once
      temp_status = 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  for (int i = 0; i < weight_blob_names.size(); ++i) { 
    // TODO(chen13) shared_weight is vector<T> but we use it as raw data including s8 type
    auto shared_weight = managed_shm_.find<vector<float, allocator<T,
         managed_shared_memory::segment_manager>>>((weight_blob_names[i] + numaId_).c_str());
    auto data_blob = ws_.GetBlob(weight_blob_names[i]);
    if (BlobIsTensorType(*data_blob, CPU)) {
      auto cpuInputTensor = data_blob->Get<Tensor>().Clone();
      cpuInputTensor.ShareExternalPointer(shared_weight.first->data());
    } else {
      auto cpuInputTensor = data_blob->template GetMutable<ideep::tensor>();
      cpuInputTensor->set_data_handle(shared_weight.first->data());
    }
  }
}

template<typename T>
int Classifier<T>::random_input_prepare(const int index) {
  Timer prepare_timer;
  prepare_timer.Start();
  if (dummy_data_) {}
  else if (batchSize_ == 1) one_idx_ = index;
  else {
    int idx = index % (iterations_ * batchSize_);
    auto offset = idx * inputSize_;
    std::memcpy(randomInputImgs_.data() + random_count_ * inputSize_,
              inputData_ + offset, sizeof(T) * inputSize_);
    randomLabels_[random_count_] = labels_[idx];
  }
  runtime_seconds_ += prepare_timer.Seconds();
  if (random_count_++ < (batchSize_ -1)) return -1;
  else random_count_ = 0;
  return 1;
}

template<typename T>
void Classifier<T>::RandomPrepareInputBlob(const vector<int>& indexes) {
  if (indexes.size() != batchSize_)
    LOG(FATAL) << "random indexes numbers not equal batch size!";
  // optimization for batchsize == 1, not memcpy for that
  if (indexes.size() == 1) {
    auto index = indexes[0] % (iterations_ * batchSize_);
    auto offset = index * inputSize_;
    PrepareInputBlob(inputData_ + offset, &(labels_[index]));
  }
#pragma omp parallel for
  for (int i = 0; i < indexes.size(); ++i) {
    auto index = indexes[i] % (iterations_ * batchSize_);
    auto offset = index * inputSize_;
    std::memcpy(randomInputImgs_.data() + random_count_ * inputSize_, inputData_ + offset, inputSize_);
    randomLabels_[i] = labels_[index];
  }
  PrepareInputBlob(randomInputImgs_.data(), randomLabels_.data());
}

template<typename T>
void Classifier<T>::PrepareInputBlob(T* inputImgs, const int* labels) {
  auto data_blob = ws_.GetBlob(input_blob_names_[0]);
  if (device_type_ == "cpu") {
    auto cpuInputTensor = data_blob->template GetMutable<Tensor>();
    ReinitializeTensor(
        cpuInputTensor,
        {batchSize_, net_conf_->channels, net_conf_->height, net_conf_->width},
        at::dtype<T>().device(CPU));
    auto input_data = cpuInputTensor->template mutable_data<T>();
    std::memcpy(input_data, inputImgs, batchSize_ * inputSize_);
  } else {
    auto inputTensor = data_blob->template GetMutable<ideep::tensor>();
    ideep::tensor::descriptor idesc;
    if (!quantized_) {
      if (dataOrder_ == "NHWC")
        LOG(FATAL) << "don't try NHWC in fp32, conv of mkldnn don't support this format";
      auto in_format = dataOrder_ == "NCHW" ? ideep::format::nchw : ideep::format::nhwc;
      idesc = ideep::tensor::descriptor(input_shape_, ideep::tensor::data_type::f32, in_format);
      inputTensor->reinit(idesc);
      inputTensor->set_data_handle(inputImgs);
    } else {
      // tansfer f32 input to s8. Int8QuantizeOp only support input datatype to be f32.
      ideep::tensor::data_type in_data_type = ideep::tensor::data_type::s8;
      auto u8_input_opt_option = getenv("U8_INPUT_OPT");
      if ((u8_input_opt_option != NULL) && (atoi(u8_input_opt_option) != 0))
        in_data_type = ideep::tensor::data_type::u8;
      auto in_format = dataOrder_ == "NCHW" ? ideep::format::nchw : ideep::format::nhwc;
      ideep::scale_t input_scale = ConvertScales({net_conf_->input_scale});
      auto odesc = ideep::tensor::descriptor(input_shape_, in_data_type, in_format);
      inputTensor->reinit(odesc);
      inputTensor->set_scale(input_scale);
      inputTensor->set_data_handle(inputImgs);
      // auto* dst_ptr = static_cast<unsigned char*>(inputTensor->get_data_handle());
      // std::stringstream output_stream;
      // output_stream << "output_data_new_" << *labels;
      // std::string output_file;
      // output_stream >> output_file;
      // print(dst_ptr, output_file, inputSize_);

      // auto asymmetric_opt_option = getenv("ASYMMETRIC_INPUT_OPT");
      // if ((asymmetric_opt_option!= NULL) && (atoi(asymmetric_opt_option) != 0)) {
      //   float iscale = 1.0/net_conf_->input_scale;
      //   float* src_ptr = static_cast<float*>(tensor.get_data_handle());
      //   char* dst_ptr = static_cast<char*>(inputTensor->get_data_handle());
      // #pragma omp parallel for collapse(3)
      //   for (int n = 0; n < tensor.get_dim(0); ++n) {
      //     for (int c = 0; c < tensor.get_dim(1); ++c) {
      //       for (int h = 0; h < tensor.get_dim(2); ++h) {
      //         for (int w = 0; w < tensor.get_dim(3); ++w) {
      //   	int idx_src = n*tensor.get_dim(1)*tensor.get_dim(2)*tensor.get_dim(3)+ 
      //   		      c*tensor.get_dim(2)*tensor.get_dim(3)+
      //   		      h*tensor.get_dim(3) + w;
      //   	int idx_dst = n*tensor.get_dim(1)*tensor.get_dim(2)*tensor.get_dim(3)+
      //   		      h*tensor.get_dim(3)*tensor.get_dim(1)+
      //   		      w*tensor.get_dim(1) + c;
      //           dst_ptr[idx_dst] = char(src_ptr[idx_src] * iscale + net_conf_->input_zero_point); 
      //         }
      //       }
      //     }
      //   }
      //   inputTensor->set_descriptor({tensor.get_dims(),
      //           ideep::tensor::data_type::u8,
      //           ideep::format::nhwc});
      // }
    }
  }
  if (accuracy_) {
    LOG(INFO) << "label layer is used!";
    auto label_blob = ws_.GetBlob(input_blob_names_[1]);
    auto labelTensor = label_blob->template GetMutable<Tensor>();
    ReinitializeTensor(
        labelTensor,
        batchSize_,
        at::dtype<int>().device(CPU));
    for (auto i = 0; i < batchSize_; ++i)
      labelTensor->template mutable_data<int>()[i] = labels[i];
  }
}

template<typename T>
static bool PairCompare(const std::pair<T, int>& lhs,
                        const std::pair<T, int>& rhs) {
  return lhs.first > rhs.first;
}

// Return the indices of the top N values of vector v.
template<typename T>
void Topk(std::vector<int>* result, std::vector<T>& v, int N) {
  std::vector<std::pair<T, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare<T>);
  for (int i = 0; i < N; ++i) {
    if (v.size() == 1) {
      result->push_back(static_cast<int>(pairs[i].first));
      return;
    }
    result->push_back(pairs[i].second);
  }
}

// in classification models, we always come across two output layers with two accuracy layer
// or one layer representing each image classification results.
template<typename T>
void Classifier<T>::AccuracyCompute(const int* labels) {
  if (output_blob_names_.size() == 1) {
    auto pos_blob = ws_.GetBlob(output_blob_names_[0]);
    int N, D;
    vector<float> pos_all;
    // output tensor may have different tensor type on different device.
    if (BlobIsTensorType(*pos_blob, CPU)) {
      auto pos_tensor = pos_blob->Get<Tensor>().Clone();
      auto meta = pos_tensor.dtype();
      N = pos_tensor.dim32(0);
      D = pos_tensor.numel() / N;
      std::stringstream meta_type;
      meta_type << meta;
      string data_type;
      meta_type >> data_type;
      if (data_type == "long") {
        auto pos_data = static_cast<long*>(pos_tensor.raw_data());
        for (int i = 0; i < N * D; ++i) pos_all.push_back(static_cast<float>(pos_data[i]));
      } else {
        auto pos_data = static_cast<T*>(pos_tensor.raw_data());
        for (int i = 0; i < N * D; ++i) pos_all.push_back(pos_data[i]);
      }
    } else {
      auto pos_tensor = pos_blob->Get<ideep::tensor>();
      N = static_cast<int>(pos_tensor.get_dim(0));
      D = static_cast<int>(pos_tensor.get_nelems() / N);
      // LOG(INFO) << "pos C is " << pos_tensor.get_dim(1) <<
      // " pos H is " << pos_tensor.get_dim(2) << " pos W is " << pos_tensor.get_dim(3);
      if (pos_tensor.get_data_type() == ideep::tensor::data_type::s32) {
        auto pos_data = static_cast<int*>(pos_tensor.get_data_handle());
        for (int i = 0; i < N * D; ++i) pos_all.push_back(static_cast<float>(pos_data[i]));
      } else if (pos_tensor.get_data_type() == ideep::tensor::data_type::f32) {
        auto pos_data = static_cast<float*>(pos_tensor.get_data_handle());
        for (int i = 0; i < N * D; ++i) pos_all.push_back(pos_data[i]);
      } else {
        auto pos_data = static_cast<T*>(pos_tensor.get_data_handle());
        for (int i = 0; i < N * D; ++i) pos_all.push_back(static_cast<float>(pos_data[i]));
      }
    }
    for (int n = 0; n < N; ++n) {
      vector<float> pos_N;
      for (int i = 0; i < D; ++i) pos_N.push_back(pos_all[D * n + i]);
      int k_nr = pos_N.size() < 5 ? 1 : 5;
      vector<int> top_k;
      Topk<float>(&top_k, pos_N, k_nr);
      // LOG(INFO) << "top 1 is " << top_k[0] -net_conf_->label_offset << " labels is " << labels[n];
      labelsOut_[n] = top_k[0] - net_conf_->label_offset;
      for (int i = 0; i < k_nr; ++i) {
        if ((top_k[0] - net_conf_->label_offset) == labels[n]) {
          top1_++;
          top5_++;
          break;
        } else if ((top_k[i] - net_conf_->label_offset) == labels[n]) {
          top5_++;
          break;
        }
      }
    }
  } else if (output_blob_names_.size() == 2) {
    auto top1_blob = ws_.GetBlob(output_blob_names_[1]);
    auto top5_blob = ws_.GetBlob(output_blob_names_[0]);
    auto top1_tensor = top1_blob->Get<Tensor>().Clone();
    auto top5_tensor = top5_blob->Get<Tensor>().Clone();
    top1_ = top1_tensor.data<float>()[0] * batchSize_ * iterations_;
    top5_ = top5_tensor.data<float>()[0] * batchSize_ * iterations_;
    LOG(INFO) << "top1 data is " << top1_tensor.data<float>()[0];
    LOG(INFO) << "top5 data is " << top5_tensor.data<float>()[0];
  }
}

template<typename T>
int Classifier<T>::run(int iteration, const bool random) {
  Timer hd_timer, prepare_timer;
  prepare_timer.Start();
  iteration = (batchSize_ == 1 ? one_idx_ : iteration);
  int idx = (iteration % iterations_) * batchSize_;
  auto offset = idx * inputSize_;
  // when dummy data is used, same iteration memory should be used
  if (dummy_data_) {}
  // if random mode batchsize == 1, you should give the iteration to real
  else if (random && batchSize_ != 1) PrepareInputBlob(randomInputImgs_.data(), randomLabels_.data());
  else PrepareInputBlob(inputData_ + offset, &(labels_[idx]));
  hd_timer.Start();
  ws_.RunNet(new_predict_net_.name());
  hd_seconds_ += hd_timer.Seconds();
  if (log_level_ == -2)
    LOG(INFO) << "net iteration " << iteration
              << " used " << hd_timer.MilliSeconds() << " ms";
  runtime_seconds_ += prepare_timer.Seconds();
  // running end of net return 1
  return 1;
}

template<typename T>
void Classifier<T>::run(const vector<int>& iterations, const bool random) {
  Timer hd_timer, prepare_timer;
  prepare_timer.Start();
  if (log_level_ == -3) hd_timer.Start();
  int idx = (iterations[0] % iterations_) * batchSize_;
  auto offset = idx * inputSize_;
  // when dummy data is used, same iteration memory should be used
  if (dummy_data_) {} 
  else if (random)  RandomPrepareInputBlob(iterations);
  else PrepareInputBlob(inputData_ + offset, &(labels_[idx]));
  hd_timer.Start();
  ws_.RunNet(new_predict_net_.name());
  hd_seconds_ += hd_timer.Seconds();
  if (log_level_ == -2)
    LOG(INFO) << "net iteration " << iterations[0]
              << " used " << hd_timer.MilliSeconds() << " ms";
  runtime_seconds_ += prepare_timer.Seconds();
}

template<typename T>
void Classifier<T>::accuracy(int iteration, const bool random) {
  iteration = (batchSize_ == 1 ? one_idx_ : iteration);
  int idx = iteration % iterations_;
  auto offset = idx * batchSize_;
  if (random && batchSize_ != 1) AccuracyCompute(randomLabels_.data());
  else AccuracyCompute(labels_.data() + offset);
}

template<typename T>
vector<int> Classifier<T>::get_labels(int iteration, const bool random) {
  accuracy(iteration, random);
  return labelsOut_;
}
template<typename T>
void Classifier<T>::getInfo(double* hd_seconds, float* top1, float* top5) {
  *top1 = top1_;
  *top5 = top5_;
  *hd_seconds = hd_seconds_;
}

template<typename T>
void Classifier<T>::getInfo(double* run_seconds, double* hd_seconds, float* top1, float* top5) {
  *top1 = top1_;
  *top5 = top5_;
  *hd_seconds = hd_seconds_;
  *run_seconds = runtime_seconds_;
}

} // using namespace caffe2
#endif // CLASSIFIER_H__
