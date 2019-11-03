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

LOADGEN_PATH ?= ../../../build/inference/loadgen
LOADGEN_INC=$(LOADGEN_PATH)
LOADGEN_FLAGS= -lpython2.7 --library :mlperf_loadgen.so --library-path $(LOADGEN_PATH)/build/lib.linux-x86_64-2.7

BUILD_DIR ?= ../../../code/resnet/int4
NVCC  ?= $(CUDA)/bin/nvcc

CUDNN_FLAGS = -L/usr/lib/x86-64-linux-gnu -I/usr/include -lcudnn
CUBLAS_FLAGS = -lcublas

.PHONY: dirs all clean

OBJS = $(BUILD_DIR)/int4_offline.a

all: dirs int4_offline

dirs:
	if [ ! -d $(BUILD_DIR) ] ; then mkdir $(BUILD_DIR)    ; fi

clean:
	rm -f int4_offline

int4_offline:
	$(NVCC) -o $@ $(OBJS) $(LOADGEN_FLAGS) $(CUDNN_FLAGS) $(CUBLAS_FLAGS)
