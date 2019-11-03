#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


sudo apt-get install -y cuda-toolkit-10.0
sudo apt-get install -y python-dev python3-dev python-pip python3-pip
sudo apt-get install -y virtualenv
sudo apt-get install -y cmake pkg-config zip g++ unzip zlib1g-dev
sudo apt-get install -y --no-install-recommends clang libglib2.0-dev
sudo apt-get install -y libhdf5-serial-dev hdf5-tools
sudo apt-get install -y zlib1g-dev zip libjpeg8-dev libhdf5-dev

# matplotlib dependencies
sudo apt-get install -y libssl-dev libfreetype6-dev libpng-dev

sudo apt-get install -y libatlas3-base
sudo apt-get install -y git
sudo apt-get install -y git-lfs && git-lfs install

cd /tmp

# install cub
wget https://github.com/NVlabs/cub/archive/1.8.0.zip -O cub-1.8.0.zip \
 && unzip cub-1.8.0.zip \
 && sudo mv cub-1.8.0/cub /usr/include/aarch64-linux-gnu/ \
 && rm -rf cub-1.8.0.zip cub-1.8.0

# install gflags
sudo rm -rf gflags \
 && git clone -b v2.2.1 https://github.com/gflags/gflags.git \
 && cd gflags \
 && mkdir build && cd build \
 && cmake -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DBUILD_gflags_LIB=ON .. \
 && make -j \
 && sudo make install \
 && cd /tmp && rm -rf gflags

# install glog
sudo rm -rf grpc \
 && git clone -b v0.3.5 https://github.com/google/glog.git \
 && cd glog \
 && cmake -H. -Bbuild -G "Unix Makefiles" -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON \
 && cmake --build build \
 && sudo cmake --build build --target install \
 && cd /tmp && rm -rf glog

# install SimpleJSON
cd /tmp \
 && rm -rf SimpleJSON \
 && git clone https://github.com/MJPA/SimpleJSON.git \
 && cd SimpleJSON \
 && mkdir build \
 && g++ -c -Wall src/JSON.cpp -o build/JSON.o \
 && g++ -c -Wall src/JSONValue.cpp -o build/JSONValue.o \
 && sudo ar rcs /usr/lib/aarch64-linux-gnu/libSimpleJSON.a build/JSON.o build/JSONValue.o \
 && sudo cp src/JSON.h /usr/include/aarch64-linux-gnu \
 && sudo cp src/JSONValue.h /usr/include/aarch64-linux-gnu \
 && cd /tmp \
 && rm -rf SimpleJSON

# Install other dependencies (PyTorch, TensorFlow, etc.)
export CUDA_ROOT=/usr/local/cuda-10.0
export CUDA_INC_DIR=$CUDA_ROOT/include
export PATH=$CUDA_ROOT/bin:$PATH
export CPATH=$CUDA_ROOT/include:$CPATH
export LIBRARY_PATH=$CUDA_ROOT/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH
wget -q https://nvidia.box.com/shared/static/veo87trfaawj5pfwuqvhl6mzc5b55fbj.whl \
     -O torch-1.1.0a0+b457266-cp36-cp36m-linux_aarch64.whl \
 && sudo python3 -m pip install -U --index-url https://pypi.org/simple --no-cache-dir setuptools==41.0.1 \
 && sudo python3 -m pip install -U numpy==1.16.4 \
    matplotlib==3.0.2 \
    grpcio==1.16.1 \
    absl-py==0.7.1 \
    py-cpuinfo==5.0.0 \
    psutil==5.6.2 \
    portpicker==1.3.1 \
    grpcio==1.16.1 \
    six==1.12.0 \
    mock==3.0.5 \
    requests==2.22.0 \
    gast==0.2.2 \
    h5py==2.10.0 \
    astor==0.8.0 \
    termcolor==1.1.0 \
    pytest==5.1.2 \
    pillow==6.0.0 \
 && sudo python3 -m pip install protobuf==3.6.1 \
    keras-preprocessing==1.0.5 \
    tensorflow-estimator==1.13.0 \
    tensorboard==1.13.0 \
    keras-applications==1.0.6 \
 && sudo python3 -m pip install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu==1.13.1+nv19.5 \
 && sudo python3 -m pip install torch-1.1.0a0+b457266-cp36-cp36m-linux_aarch64.whl \
 && sudo python3 -m pip install torchvision==0.2.2.post3 \
 && sudo -E python3 -m pip install pycuda==2019.1 \
 && sudo python3 -m pip install Cython==0.29.10 \
 && sudo python3 -m pip install pycocotools==2.0.0 \
 && rm torch-1.1.0a0+b457266-cp36-cp36m-linux_aarch64.whl \
 && sudo python -m pip install absl-py==0.7.1
