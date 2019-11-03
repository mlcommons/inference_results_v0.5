#ifndef MISC_H__
#define MISC_H__
#include <fstream>
#include <string>
#include <vector>
#include <dirent.h>
#include "caffe2/core/blob.h"
#include "caffe2/ideep/ideep_utils.h"

namespace caffe2 {

template<typename T>
void print(const T* data, const std::string& name, size_t size = 1000) {
  // print output file
  auto pos = name.rfind("/");
   
  string output_file = (pos != string::npos ? name.substr(pos + 1) : name) + ".txt";
  std::ofstream output_data(output_file);
  for (size_t i = 0; i < size; ++i) {
    output_data << static_cast<float>(data[i]) << "\n";
  }
  output_data.close();
}
void print(const Blob *blob, const std::string &name, size_t size = 1000, string device_type = "ideep") {
  const float* data;
  if (device_type == "ideep") {
    auto tensor = blob->Get<ideep::tensor>();
    data = static_cast<float *>(tensor.get_data_handle());
  } else {
    auto tensor = blob->Get<Tensor>().Clone();
    data = tensor.data<float>();
  }
  // print output file
  string output_file = name + ".txt";
  std::ofstream output_data(output_file);
  for (size_t i = 0; i < size; ++i) {
    output_data << data[i] << "\n";
  }
  output_data.close();
}

void stringSplit(vector<string>* split_list, const string& str_list, const string& split_op){
  std::string::size_type pos1 = 0;
  std::string::size_type pos2 = str_list.find(split_op);
  while(std::string::npos != pos2) {
    (*split_list).push_back(str_list.substr(pos1, pos2));
    pos1 = pos2 + split_op.size();
    pos2 = str_list.find(split_op, pos1);
  }
  if (pos1 != str_list.size()) (*split_list).push_back(str_list.substr(pos1));
}

} // using namespace caffe2
#endif // MISC_H__
