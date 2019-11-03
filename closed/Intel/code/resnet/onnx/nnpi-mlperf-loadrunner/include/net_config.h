#ifndef __NETCONF_H__
#define __NETCONF_H__
#include <string>
#include <vector>
// we don't open yaml support util
// you need to change your net configure file
#define USE_YAML 0

using namespace std;

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
  float input_scale = 1.18944883347;
  int input_zero_point = 0;
  vector<float> mean_value = {103.94, 116.78, 123.68};

  NetConf() {}
  NetConf(const string& net_conf) {
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
    input_scale = 1.18944883347;
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
    input_scale = 0.0078740157187;
    mean_value = {127.5, 127.5, 127.5};
  }
};

std::unique_ptr<NetConf> get_net_conf(const string& net_conf) {
  if(net_conf == "resnet50")
    return std::unique_ptr<Resnet50Conf>(new Resnet50Conf());
  else if (net_conf == "mobilenetv1")
    return std::unique_ptr<Mobilenetv1Conf>(new Mobilenetv1Conf());
  else
    cout << "specified net configuration should enable yaml-cpp"
               << "or you can just input 'resnet50' 'mobilenetv1' as net_conf";
}

#endif // NETCONF_H__
