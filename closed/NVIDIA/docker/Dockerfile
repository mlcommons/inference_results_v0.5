# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt:19.09-py3
FROM ${BASE_IMAGE}

# Explicitly use bash instead of sh ('echo' behaves differently on some shells)
SHELL ["/bin/bash", "-c"]

RUN echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/ /" > /etc/apt/sources.list.d/cuda.list \
 && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub \
 && apt-get update \
 && apt-get install -y --no-install-recommends build-essential autoconf libtool git \
        curl wget pkg-config sudo ca-certificates vim-tiny automake libssl-dev \
        bc python3-dev python3-pip python2.7-dev google-perftools gdb libglib2.0-dev clang python-pip sshfs libre2-dev \
 && apt-get install -y --no-install-recommends cuda-cudart-dev-10-1 cuda-curand-dev-10-1 cuda-nvrtc-dev-10-1 cuda-command-line-tools-10-1 libcublas-dev \
 && ln -s cuda-10.1 /usr/local/cuda \
 && apt-get remove -y cmake \
 && apt-get remove -y libgflags-dev \
 && apt-get remove -y libprotobuf-dev \
 && apt-get -y autoremove \
 && rm -rf /var/lib/apt/lists/*

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

### Install necessary python 2 modules
RUN python2 -m pip install --upgrade pip \
 && python2 -m pip install --upgrade setuptools wheel virtualenv \
 && python2 -m pip install ez_setup==0.9 \
 && python2 -m pip install numpy \
 && python2 -m pip install absl-py==0.7.1

RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install --upgrade setuptools wheel virtualenv \
 && python3 -m pip install cmake==3.11.0 \
 && python3 -m pip install pillow==6.0.0 \
 && python3 -m pip install opencv-python-headless

# install gflags
# -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DBUILD_gflags_LIB=ON .. \
RUN git clone -b v2.2.1 https://github.com/gflags/gflags.git \
 && cd gflags \
 && mkdir build && cd build \
 && cmake -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DBUILD_gflags_LIB=ON .. \
 && make -j \
 && make install \
 && cd /tmp && rm -rf gflags

# install glog
RUN git clone -b v0.3.5 https://github.com/google/glog.git \
 && cd glog \
 && cmake -H. -Bbuild -G "Unix Makefiles" -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON \
 && cmake --build build \
 && cmake --build build --target install \
 && cd /tmp && rm -rf glog

WORKDIR /tmp
RUN wget https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh \
 && chmod +x wait-for-it.sh \
 && mv wait-for-it.sh /usr/local/bin/

RUN git clone -b v1.0.6 https://github.com/dcdillon/cpuaff \
 && cd cpuaff \
 && ls -lF \
 && ./bootstrap.sh \
 && ./configure \
 && make \
 && make install \
 && cd ../ \
 &&  rm -rf cpuaff

RUN git clone -b v1.4.1 https://github.com/google/benchmark.git \
 && cd benchmark \
 && git clone -b release-1.8.0 https://github.com/google/googletest.git \
 && mkdir build && cd build \
 && cmake .. -DCMAKE_BUILD_TYPE=RELEASE \
 && make -j && make install \
 && cd /tmp && rm -rf benchmark

RUN git clone https://github.com/jupp0r/prometheus-cpp.git \
 && cd prometheus-cpp \
 && git checkout -b yais e7709f7e3b71bc5b1ac147971c87f2f0ae9ea358 \
 && git submodule update --init --recursive \
 && mkdir build && cd build \
 && cmake -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE .. \
 && make -j \
 && make install

RUN wget https://dl.influxdata.com/telegraf/releases/telegraf-1.7.1-static_linux_amd64.tar.gz \
 && tar xzf telegraf-1.7.1-static_linux_amd64.tar.gz \
 && mv telegraf/telegraf /usr/local/bin \
 && rm -rf telegraf-1.7.1-static_linux_amd64.tar.gz telegraf

RUN git clone https://github.com/cameron314/concurrentqueue.git \
 && cd concurrentqueue \
 && git checkout 8f65a87 \
 && mkdir -p /usr/local/include/moodycamel \
 && cp *.h /usr/local/include/moodycamel/

RUN git clone https://github.com/bloomen/transwarp.git \
 && cd transwarp \
 && git checkout 1.8.0 \
 && mkdir -p /usr/local/include/transwarp \
 && cp src/transwarp.h /usr/local/include/transwarp/transwarp.h \
 && cd .. && rm -rf transwarp

# install flatbuffers
RUN git clone -b v1.10.0 https://github.com/google/flatbuffers.git \
 && cd flatbuffers \
 && mkdir build2 && cd build2 \
 && cmake -DCMAKE_BUILD_TYPE=Release .. \
 && make -j$(nproc) install \
 && rm -rf /flatbuffers

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config zip g++ zlib1g-dev unzip python \
 && rm -rf /var/lib/apt/lists/*

ENV BAZEL_VERSION="0.21.0"

RUN wget https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh \
 && chmod +x bazel-$BAZEL_VERSION-installer-linux-x86_64.sh \
 && ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh \
 && rm -f bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Install dependencies of TensorRT-Laboratory
RUN python3 -m pip install click==6.7 \
 && python3 -m pip install Jinja2==2.10 \
 && python3 -m pip install MarkupSafe==1.0 \
 && python3 -m pip install grpcio==1.16.1 \
 && python3 -m pip install matplotlib==3.0.2 \
 && python3 -m pip install onnx==1.3.0 \
 && python3 -m pip install jupyter-client==5.2.4 \
 && python3 -m pip install jupyter-core==4.4.0 \
 && python3 -m pip install jupyterlab==0.35.4 \
 && python3 -m pip install jupyterlab-server==0.2.0 \
 && python3 -m pip install wurlitzer==1.0.2 \
 && python3 -m pip install pytest==4.6.2

# Install TensorFlow, needed by SSDMobileNet benchmark
# Install CPU version since we don't actually need to run TensorFlow.
RUN python3 -m pip install tensorflow==1.13.1

# Install pytorch and torchvision, needed by SSDResNet34 benchmark
RUN python3 -m pip install torch==1.1.0 \
 && python3 -m pip install torchvision==0.3.0 \
 && python3 -m pip install pycuda==2019.1 \
 && python3 -m pip install Cython==0.29.10 \
 && python3 -m pip install pycocotools==2.0.0

# Install sacrebleu, needed by GNMT benchmark
RUN python3 -m pip install sacrebleu==1.3.3

# Install CUB, needed by GNMT benchmark
RUN wget https://github.com/NVlabs/cub/archive/1.8.0.zip -O cub-1.8.0.zip \
 && unzip cub-1.8.0.zip \
 && mv cub-1.8.0/cub /usr/include/x86_64-linux-gnu/ \
 && rm -rf cub-1.8.0.zip cub-1.8.0

# Install TensorRT Python bindings
RUN bash /opt/tensorrt/python/python_setup.sh

#install libjemalloc2
RUN echo 'deb http://archive.ubuntu.com/ubuntu disco main restricted universe multiverse' | tee -a /etc/apt/sources.list.d/disco.list \
  && echo 'Package: *\nPin: release a=disco\nPin-Priority: -10\n' | tee -a /etc/apt/preferences.d/disco.pref \
  && apt-get update \
  && apt-get install --no-install-recommends -t disco -y libjemalloc2 libtcmalloc-minimal4

# Install TRT-Lab, which requires g++-5
RUN apt-get update && apt-get install -y --no-install-recommends g++-5 \
 && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 60 --slave /usr/bin/g++ g++ /usr/bin/g++-5 \
 && git clone https://github.com/NVIDIA/tensorrt-laboratory.git \
 && cd tensorrt-laboratory \
 && git checkout e1eed23d1966fdbe8b2c9d2f5e8e1da5b17b94f8 \
 && mkdir -p build \
 && cd build \
 && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_NVRPC=OFF -DENABLE_TESTING=OFF .. \
 && make -j \
 && make install \
 && cd ../.. \
 && rm -rf tensorrt-laboratory

# Instsall SimpleJSON
RUN git clone https://github.com/MJPA/SimpleJSON.git \
 && cd SimpleJSON \
 && mkdir build \
 && g++ -c -Wall src/JSON.cpp -o build/JSON.o \
 && g++ -c -Wall src/JSONValue.cpp -o build/JSONValue.o \
 && ar rcs /usr/lib/x86_64-linux-gnu/libSimpleJSON.a build/JSON.o build/JSONValue.o \
 && cp src/JSON.h /usr/include/x86_64-linux-gnu \
 && cp src/JSONValue.h /usr/include/x86_64-linux-gnu \
 && cd .. \
 && rm -rf SimpleJSON

# Copy repo to work directory
COPY . /work

# Build binaries so that the image can be used out-of-the-box
RUN cd /work && make build

WORKDIR /work
