# MLPerf Inference HanGuangAI C++ Implementation Instruction for resnet/Offline

## Environment Setup

### Python

#### Install Python 3 Virtual Environment

   ```shell
   $ python3 -m venv py3
   ```

#### Activate Python 3 Virtual Environment

   ```shell
   $ source py3/bin/activate
   ```

#### Install Python Requisites

   ```shell
   $ pip install --upgrade pip
   $ pip install --upgrade setuptools
   $ pip install scipy
   $ pip install six numpy wheel setuptools mock 'future>=0.17.1'
   $ pip install keras_applications==1.0.6 --no-deps
   $ pip install keras_preprocessing==1.0.5 --no-deps
   $ pip install pudb  
   ```

### Tensorflow 1.13.1

#### Download Source Code
Click [here](https://github.com/tensorflow/tensorflow/archive/v1.13.1.tar.gz) to download tensorflow official release 1.13.1 source code package.

#### Verfiy Bazel version is 0.19.0 (which works with tensorflow 1.13.1)

   ```shell
   bazel version
   ```

#### Build and Install Tensorflow from Source Code

   ```shell
   cd tensorflow-1.13.1
   ./configure
   bazel build -c opt --copt=-msse4.2 --copt=-mavx2 --copt=-mfma //tensorflow/tools/pip_package:build_pip_package
   bazel-bin/tensorflow/tools/pip_package/build_pip_package ../
   pip install ../tensorflow-1.13.1-cp35-cp35m-linux_x86_64.whl
   ```

#### Build Tensorflow CC

   ```shell
   $ bazel build -c opt --copt=-msse4.2 --copt=-mavx2 --copt=-mfma //tensorflow:libtensorflow_cc.so
   ```

### MLPerf Inference

#### Clone the source code

   ```shell
   git clone --recursive git@github.com:mlperf/inference.git 
   ```
#### Download the resnet model

As indicated [here](https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection#supported-models) in MLPerf inference source code. Rename the model pb file to fp32.pb for quantization usage.

#### Download the imagenet dataset

As indicated [here](https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection#datasets) in MLPerf inference source code

### HanGuangAI

* HanGuangAI is a software package developed by T-Head (PingTouGe) Semiconductor Co., Ltd., subsidiary of Alibaba Group Holding Ltd., connecting popluar frameworks to HanGuang NPU. It will be distributed to customer once HanGuang NPU is made public accessible. 
* Install HanGuang AI 1.0.3
  
  ```shell
  pip install hgai-ubuntu-1.0.3-cp35-cp35m-linux_x86_64.whl
  ```

### Environment Variables
* Add tensorflow cc library location to LD_LIBRARY_PATH
* Add path environment variables used by c++ test harness build

   ```shell
   $ export LG_PATH=/path/to/inference/loadgen
   $ export TF_PATH=/path/to/tensorflow/source
   $ export MODEL_DIR=/path/to/resnet_model/
   $ export DATA_DIR=/path/to/dataset/
   ```

### Install OpenCV
   ```shell
   $ sudo apt install libopencv-dev
   ```

### Build the C++ Test Harness

   ```shell
   $ cd code/resnet/tfcpp/classification/cpp
   $ mkdir build
   $ cd build
   $ cmake ..
   $ make
   ```

## Quantization

   ```shell
   $ cd code/resnet/tfcpp/quantize
   $ python converter.py --output_type npu
   ```

## Execution

### Accuracy Mode

   ```shell
   $ cd code/resnet/tfcpp/classification/cpp
   $ ./run_accuracy.sh Offline
   ```
   
### Performance Mode

   ```shell
   $ cd code/resnet/tfcpp/classification/cpp
   $ ./run_perf.sh Offline
   ```
  
### Submission Mode

   ```shell
   $ cd code/resnet/tfcpp/classification/cpp
   $ ./run.sh Offline
   ```
