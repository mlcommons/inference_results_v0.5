# Caffe2 c++ inference benchmark for MKL-DNN
This is a c++ code for inference benchmark test for MKL-DNN.

## Dependencies
caffe2:
Download pytorch and build with MKL-DNN
please refer https://github.com/pytorch/pytorch
protobuf:
https://github.com/google/protobuf/blob/master/src/README.md
opencv:
https://docs.opencv.org/master/df/d65/tutorial_table_of_content_introduction.html
yaml-cpp:
https://github.com/jbeder/yaml-cpp.git
when use yaml-cpp make sure to cmake like below
CC=$(which gcc) CXX=$(which g++) cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/home/to/your/dir ..



## Build and run

```bash
make
./test_tools.sh
```
edit Makefile for build benchmark_inference_mkldnn app:
CAFFE2_DIR is the pytorch code path, you should set it both in Makefile and test_tools.sh of the path
MKL_DIR is the mkl library path
LDFLAGS is third party library

This folder include Makefile for example.

edit test_tools.sh for run the app:
batchsize_i is the batch sizes you want to input
iteration_i is number of iterations
net_name_i is the net name for classification, now support resnet50
data_type_i is the data type to use, now support fp32 and int8
device_type_i is the device type, default ideep
output_type is the output operator's running device, eg resnet50's last layer is argmax which
is runnig on cpu instead of ideep 
file_list_i is the validation data path
LD_PRELOAD is iomp library path (it is not necessary)
LD_LIBRARY_PATH is caffe2 library path

net_config.h is the file for net config, when new net added you should add net config info into this file.
