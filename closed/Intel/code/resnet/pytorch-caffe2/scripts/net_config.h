#ifndef NETCONF_H__
#define NETCONF_H__
#include <string>
#include <vector>
// we don't open yaml support util
// you need to change your net configure file
#define USE_YAML 0
#if USE_YAML
#include "yaml-cpp/yaml.h"
#endif

namespace caffe2 {

class NetConf {
 public:
  string net_name = "";
  string preprocess_method = "ResizeWithAspect";
  bool use_accuracy_layer = false;
  bool bgr2rgb = true;
  int channels = 3;
  int label_offset = 1;
  int height = 224;
  int width = 224;
  int rescale_size = 256;
  float scale = 1;
  float aspect_scale = 87.5;
  double input_scale = 1.18944883347;
  int input_zero_point = 0;
  vector<float> mean_value = {103.94, 116.78, 123.68};

  NetConf() {}
  NetConf(const string& net_conf) {
#if USE_YAML
    if (!net_conf.empty()) {
      YAML::Node net = YAML::LoadFile(net_conf);
      net_name = net["net_name"].as<string>();
      preprocess_method = net["preprocess_method"].as<string>();
      use_accuracy_layer = net["use_accuracy_layer"].as<bool>();
      bgr2rgb = net["bgr2rgb"].as<bool>();
      channels = net["channels"].as<int>();
      label_offset = net["label_offset"].as<int>();
      height = net["height"].as<int>();
      width = net["width"].as<int>();
      rescale_size = net["rescale_size"].as<int>();
      scale = net["scale"].as<float>();
      aspect_scale = net["aspect_scale"].as<float>();
      input_scale = net["input_scale"].as<double>();
      mean_value = net["mean_value"].as<vector<float>>();
    }
#endif
  }
};

class Resnet50Conf : public NetConf {
 public:
  Resnet50Conf() {
    net_name = "resnet50";
    preprocess_method = "ResizeWithAspect";
    use_accuracy_layer = false;
    bgr2rgb = true;
    channels = 3;
    label_offset = 1;
    height = 224;
    width = 224;
    rescale_size = 256;
    scale = 1;
    aspect_scale = 87.5;
    input_scale = 1.1894488334655762; 
    auto asymmetric_opt_option = getenv("ASYMMETRIC_INPUT_OPT");
    if ((asymmetric_opt_option!= NULL) && (atoi(asymmetric_opt_option) != 0)) {
      input_scale = 1.07741177082;
      input_zero_point = 114;
    } 
    mean_value = {103.94, 116.78, 123.68};
  }
};

class Mobilenetv1Conf : public NetConf {
 public:
  Mobilenetv1Conf() {
    net_name = "mobilenetv1";
    preprocess_method = "ResizeWithAspect";
    use_accuracy_layer = false;
    bgr2rgb = true;
    channels = 3;
    label_offset = 1;
    height = 224;
    width = 224;
    rescale_size = 256;
    scale = 0.00790489464998;
    aspect_scale = 87.5;
    input_scale = 0.007874015718698502;
    mean_value = {127.5, 127.5, 127.5};
  }
};

std::unique_ptr<NetConf> get_net_conf(const string& net_conf) {
#if USE_YAML
  return std::unique_ptr<NetConf>(new NetConf(net_conf));
#else
  if(net_conf == "resnet50")
    return std::unique_ptr<Resnet50Conf>(new Resnet50Conf());
  else if (net_conf == "mobilenetv1")
    return std::unique_ptr<Mobilenetv1Conf>(new Mobilenetv1Conf());
  else
    LOG(FATAL) << "specified net configuration should enable yaml-cpp"
               << "or you can just input 'resnet50' 'mobilenetv1' as net_conf";
#endif
}

} // use namespace caffe2
#endif // NETCONF_H__
