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

SHELL := /bin/bash
MAKEFILE_NAME := $(lastword $(MAKEFILE_LIST))

ARCH := $(shell uname -p)
UNAME := $(shell whoami)
UID := $(shell id -u `whoami`)
GROUPNAME := $(shell id -gn `whoami`)
GROUPID := $(shell id -g `whoami`)

BENCHMARK_LIST := GNMT MobileNet ResNet50 SSDMobileNet SSDResNet34

# Conditional Docker flags
ifndef DOCKER_DETACH
DOCKER_DETACH := 0
endif
ifndef DOCKER_TAG
DOCKER_TAG := $(UNAME)
endif

PROJECT_ROOT := $(shell pwd)
BUILD_DIR    := $(PROJECT_ROOT)/build

# CuDNN and TensorRT Bindings
ifeq ($(ARCH), aarch64)
CUDA_VER   := 10.0
TRT_VER    := 6.0.1.5
UBUNTU_VER := 18.04
CUDNN_VER  := 7.6
CUB_VER    := 1.8.0
else
CUDA_VER   := 10.1
TRT_VER    := 6.0.1.5
UBUNTU_VER := 16.04
CUDNN_VER  := 7.6
CUB_VER    := 1.8.0
endif

# Set the include directory for Loadgen header files
INFERENCE_DIR = $(BUILD_DIR)/inference
LOADGEN_INCLUDE_DIR := $(INFERENCE_DIR)/loadgen
INFERENCE_HASH = 61220457dec221ed1984c62bd9d382698bd71bc6

# Set Environment variables to extracted contents
export LD_LIBRARY_PATH := /usr/local/cuda-$(CUDA_VER)/lib64:/usr/lib/$(ARCH)-linux-gnu:$(LOADGEN_INCLUDE_DIR)/build/lib.linux-x86_64-2.7:$(LD_LIBRARY_PATH)
export LIBRARY_PATH := /usr/local/cuda-$(CUDA_VER)/lib64:/usr/lib/$(ARCH)-linux-gnu:$(LOADGEN_INCLUDE_DIR)/build/lib.linux-x86_64-2.7:$(LIBRARY_PATH)
export PATH := /usr/local/cuda-$(CUDA_VER)/bin:$(PATH)
export CPATH := /usr/local/cuda-$(CUDA_VER)/include:/usr/include/$(ARCH)-linux-gnu:/usr/include/$(ARCH)-linux-gnu/cub:$(CPATH)
export CUDA_PATH := /usr/local/cuda-$(CUDA_VER)

# Set CUDA_DEVICE_MAX_CONNECTIONS to increase multi-stream performance.
export CUDA_DEVICE_MAX_CONNECTIONS := 32

# Set DATA_DIR, PREPROCESSED_DATA_DIR, and MODEL_DIR if they are not already set
ifndef DATA_DIR
	export DATA_DIR := $(BUILD_DIR)/data
endif
ifndef PREPROCESSED_DATA_DIR
	export PREPROCESSED_DATA_DIR := $(BUILD_DIR)/preprocessed_data
endif
ifndef MODEL_DIR
	export MODEL_DIR := $(BUILD_DIR)/models
endif

# Set path to dataset and preprocessed data in datacenter.
MLPERF_INFERENCE_PATH :=
# ComputeLab
ifneq ($(wildcard /home/scratch.mlperf_inference),)
	MLPERF_INFERENCE_PATH := /home/scratch.mlperf_inference
endif
# Toto
ifneq ($(wildcard /scratch/datasets/mlperf_inference),)
	MLPERF_INFERENCE_PATH := /scratch/datasets/mlperf_inference
endif
# Circe
ifneq ($(wildcard /gpfs/fs1/datasets/mlperf_inference),)
	MLPERF_INFERENCE_PATH := /gpfs/fs1/datasets/mlperf_inference
endif

# Specify default dir for harness output logs.
ifndef LOG_DIR
	export LOG_DIR := $(BUILD_DIR)/logs/$(shell date +'%Y.%m.%d-%H.%M.%S')
endif

# Specify debug options for build (default to Release build)
ifeq ($(DEBUG),1)
BUILD_TYPE := Debug
LOADGEN_DEBUG_BUILD := true
else
BUILD_TYPE := Release
LOADGEN_DEBUG_BUILD := false
endif

############################## PREBUILD ##############################
.PHONY: prebuild
prebuild:
	@$(MAKE) -f $(MAKEFILE_NAME) build_docker
ifneq ($(strip ${DOCKER_DETACH}), 1)
	@$(MAKE) -f $(MAKEFILE_NAME) attach_docker
endif

# Add symbolic links to preprocessed datasets if they exist.
.PHONY: link_dataset_dir
link_dataset_dir:
	@mkdir -p build
ifneq ($(MLPERF_INFERENCE_PATH),)
	@if [ ! -e $(DATA_DIR) ]; then \
		ln -sn $(MLPERF_INFERENCE_PATH)/data $(DATA_DIR); \
	fi
	@if [ ! -e $(PREPROCESSED_DATA_DIR) ]; then \
		ln -sn $(MLPERF_INFERENCE_PATH)/preprocessed_data $(PREPROCESSED_DATA_DIR); \
	fi
	@if [ ! -e $(MODEL_DIR) ]; then \
		ln -sn $(MLPERF_INFERENCE_PATH)/models $(MODEL_DIR); \
	fi
endif

.PHONY: build_docker
build_docker: link_dataset_dir
ifeq ($(ARCH), x86_64)
	@echo "Building Docker image"
	docker build -t mlperf-inference:$(DOCKER_TAG)-latest \
		--network host -f docker/Dockerfile .
endif

.PHONY: docker_add_user
docker_add_user:
ifeq ($(ARCH), x86_64)
	@echo "Adding user account into image"
	docker build -t mlperf-inference:$(DOCKER_TAG) --network host \
		--build-arg BASE_IMAGE=mlperf-inference:$(DOCKER_TAG)-latest \
		--build-arg GID=$(GROUPID) --build-arg UID=$(UID) --build-arg GROUP=$(GROUPNAME) --build-arg USER=$(UNAME) \
		- < docker/Dockerfile.user
endif

.PHONY: attach_docker
attach_docker: docker_add_user
ifeq ($(ARCH), x86_64)
	@echo "Launching Docker interactive session"
	nvidia-docker run --rm -ti -w /work -v ${PWD}:/work -v ${HOME}:/mnt/${HOME} \
		-v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro \
		--security-opt apparmor=unconfined --security-opt seccomp=unconfined \
		--name mlperf-inference-$(UNAME) -h mlperf-inference-$(UNAME) --add-host mlperf-inference-$(UNAME):127.0.0.1 \
		`if [ -d /home/scratch.mlperf_inference ]; then echo "-v /home/scratch.mlperf_inference:/home/scratch.mlperf_inference"; fi` \
		`if [ -d /scratch/datasets/mlperf_inference ]; then echo "-v /scratch/datasets/mlperf_inference:/scratch/datasets/mlperf_inference"; fi` \
		`if [ -d /gpfs/fs1/datasets/mlperf_inference ]; then echo "-v /gpfs/fs1/datasets/mlperf_inference:/gpfs/fs1/datasets/mlperf_inference"; fi` \
		--user $(UID):$(GROUPID) --net host --device /dev/fuse --cap-add SYS_ADMIN mlperf-inference:$(DOCKER_TAG)
endif

.PHONY: download_dataset
download_dataset:
	@mkdir -p $(DATA_DIR)/imagenet
	@wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar -O $(DATA_DIR)/imagenet/ILSVRC2012_img_val.tar
	@cd $(DATA_DIR)/imagenet && tar -xf ILSVRC2012_img_val.tar

.PHONY: preprocess_data
preprocess_data:
	@python3 scripts/preprocess_data.py -d $(DATA_DIR) -o build/preprocessed_data

INT4_DIR := $(PROJECT_ROOT)/code/resnet/int4

############################### BUILD ###############################
.PHONY: build
build: clone_loadgen link_dataset_dir
ifeq ($(ARCH), x86_64)
	@$(MAKE) -f $(MAKEFILE_NAME) build_loadgen_so
	@$(MAKE) -f $(MAKEFILE_NAME) build_int4
else
	@$(MAKE) -f $(MAKEFILE_NAME) build_loadgen
	@$(MAKE) -f $(MAKEFILE_NAME) build_gnmt
endif

# Clone LoadGen repo.
.PHONY: clone_loadgen
clone_loadgen:
	@if [ ! -d $(LOADGEN_INCLUDE_DIR) ]; then \
		echo "Cloning Official MLPerf Inference (For Loadgen Files)" \
			&& git clone https://github.com/mlperf/inference.git $(INFERENCE_DIR) \
			&& cd $(INFERENCE_DIR) \
			&& git checkout $(INFERENCE_HASH) \
			&& git submodule update --init build \
			&& git submodule update --init third_party/gn \
			&& git submodule update --init third_party/ninja \
			&& git submodule update --init third_party/pybind; \
	fi

.PHONY: build_loadgen_so
build_loadgen_so:
ifeq ($(ARCH), x86_64)
	@echo "Building mlperf-inference.so" \
		&& cd $(LOADGEN_INCLUDE_DIR) \
		&& CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel
endif

.PHONY: build_int4
build_int4:
ifeq ($(ARCH), x86_64)
	@echo "Building INT4 harness..."
	cd $(INT4_DIR) \
		&& make -j CUDA=$(CUDA_PATH) BUILD_DIR=$(INT4_DIR) LOADGEN_PATH=$(LOADGEN_INCLUDE_DIR) clean  \
		&& make -j CUDA=$(CUDA_PATH) BUILD_DIR=$(INT4_DIR) LOADGEN_PATH=$(LOADGEN_INCLUDE_DIR) all
endif

# Build LoadGen.
.PHONY: build_loadgen
build_loadgen:
ifneq ($(ARCH), x86_64)
	@echo "Building loadgen..."
	@cd $(INFERENCE_DIR)/third_party/ninja \
		&& python2 configure.py --bootstrap \
		&& cd ../.. \
		&& python2 third_party/gn/build/gen.py \
		&& third_party/ninja/ninja -C third_party/gn/out \
		&& cp third_party/gn/out/gn* third_party/gn/. \
		&& third_party/gn/gn gen out/MakefileGnProj --args="is_debug=$(LOADGEN_DEBUG_BUILD)" \
		&& third_party/ninja/ninja -C out/MakefileGnProj mlperf_loadgen
endif

# Build GNMT source codes.
.PHONY: build_gnmt
build_gnmt:
ifneq ($(ARCH), x86_64)
	@echo "Building GNMT source..."
	cd code/gnmt/tensorrt/src \
		&& make -j
	@echo "Building GNMT harness..."
	@mkdir -p build/harness \
		&& cd build/harness \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DLOADGEN_INCLUDE_DIR=$(LOADGEN_INCLUDE_DIR) $(PROJECT_ROOT)/code/gnmt/tensorrt/harness \
		&& make -j
endif

###############################  RUN  ###############################
.PHONY: run_intx_test_harness
run_intx_test_harness: link_dataset_dir
	@mkdir -p $(LOG_DIR)/ResNet50_int4
	@cd $(INT4_DIR) && ./int4_offline $(INT4_ARGS) --lgls_logfile_outdir $(LOG_DIR)/ResNet50_int4
	@if [ `stat -c "%s" $(LOG_DIR)/ResNet50_int4/mlperf_log_accuracy.json` -gt 4 ]; then \
		python3 $(INFERENCE_DIR)/v0.5/classification_and_detection/tools/accuracy-imagenet.py --mlperf-accuracy-file $(LOG_DIR)/ResNet50_int4/mlperf_log_accuracy.json --imagenet-val-file data_maps/imagenet/val_map.txt --dtype int32 ; \
	fi

.PHONY: run_int4_TitanRTXx4_accuracy
run_int4_TitanRTXx4_accuracy: link_dataset_dir
	@mkdir -p $(LOG_DIR)/TitanRTXx4/resnet/Offline/accuracy
	@cd $(INT4_DIR) && ./int4_offline -b 1024 -a autoconfig_rtx --mlperf_conf_path $(PROJECT_ROOT)/measurements/TitanRTXx4/resnet/Offline/mlperf.conf --user_conf_path $(PROJECT_ROOT)/measurements/TitanRTXx4/resnet/Offline/user.conf --test-mode AccuracyOnly --lgls_logfile_outdir $(LOG_DIR)/TitanRTXx4/resnet/Offline/accuracy
	@python3 $(INFERENCE_DIR)/v0.5/classification_and_detection/tools/accuracy-imagenet.py --mlperf-accuracy-file $(LOG_DIR)/TitanRTXx4/resnet/Offline/accuracy/mlperf_log_accuracy.json --imagenet-val-file data_maps/imagenet/val_map.txt --dtype int32 2>&1 | tee $(LOG_DIR)/TitanRTXx4/resnet/Offline/accuracy/accuracy.txt

.PHONY: run_int4_TitanRTXx4_performance
run_int4_TitanRTXx4_performance: link_dataset_dir
	@mkdir -p $(LOG_DIR)/TitanRTXx4/resnet/Offline/performance/run_1
	@cd $(INT4_DIR) && ./int4_offline -b 1024 -a autoconfig_rtx --mlperf_conf_path $(PROJECT_ROOT)/measurements/TitanRTXx4/resnet/Offline/mlperf.conf --user_conf_path $(PROJECT_ROOT)/measurements/TitanRTXx4/resnet/Offline/user.conf --test-mode PerformanceOnly --lgls_logfile_outdir $(LOG_DIR)/TitanRTXx4/resnet/Offline/performance/run_1

.PHONY: run_int4_T4x8_accuracy
run_int4_T4x8_accuracy: link_dataset_dir
	@mkdir -p $(LOG_DIR)/T4x8/resnet/Offline/accuracy
	@cd $(INT4_DIR) && ./int4_offline -b 512 -a autoconfig_t4 --mlperf_conf_path $(PROJECT_ROOT)/measurements/T4x8/resnet/Offline/mlperf.conf --user_conf_path $(PROJECT_ROOT)/measurements/T4x8/resnet/Offline/user.conf --test-mode AccuracyOnly --lgls_logfile_outdir $(LOG_DIR)/T4x8/resnet/Offline/accuracy
	@python3 $(INFERENCE_DIR)/v0.5/classification_and_detection/tools/accuracy-imagenet.py --mlperf-accuracy-file $(LOG_DIR)/T4x8/resnet/Offline/accuracy/mlperf_log_accuracy.json --imagenet-val-file data_maps/imagenet/val_map.txt --dtype int32 2>&1 | tee $(LOG_DIR)/T4x8/resnet/Offline/accuracy/accuracy.txt

.PHONY: run_int4_T4x8_performance
run_int4_T4x8_performance: link_dataset_dir
	@mkdir -p $(LOG_DIR)/T4x8/resnet/Offline/performance/run_1
	@cd $(INT4_DIR) && ./int4_offline -b 512 -a autoconfig_t4 --mlperf_conf_path $(PROJECT_ROOT)/measurements/T4x8/resnet/Offline/mlperf.conf --user_conf_path $(PROJECT_ROOT)/measurements/T4x8/resnet/Offline/user.conf --test-mode PerformanceOnly --lgls_logfile_outdir $(LOG_DIR)/T4x8/resnet/Offline/performance/run_1

.PHONY: run_int4_T4x20_accuracy
run_int4_T4x20_accuracy: link_dataset_dir
	@mkdir -p $(LOG_DIR)/T4x20/resnet/Offline/accuracy
	@cd $(INT4_DIR) && ./int4_offline -b 512 -a autoconfig_t4 --mlperf_conf_path $(PROJECT_ROOT)/measurements/T4x20/resnet/Offline/mlperf.conf --user_conf_path $(PROJECT_ROOT)/measurements/T4x20/resnet/Offline/user.conf --test-mode AccuracyOnly --lgls_logfile_outdir $(LOG_DIR)/T4x20/resnet/Offline/accuracy
	@python3 $(INFERENCE_DIR)/v0.5/classification_and_detection/tools/accuracy-imagenet.py --mlperf-accuracy-file $(LOG_DIR)/T4x20/resnet/Offline/accuracy/mlperf_log_accuracy.json --imagenet-val-file data_maps/imagenet/val_map.txt --dtype int32 2>&1 | tee $(LOG_DIR)/T4x20/resnet/Offline/accuracy/accuracy.txt

.PHONY: run_int4_T4x20_performance
run_int4_T4x20_performance: link_dataset_dir
	@mkdir -p $(LOG_DIR)/T4x20/resnet/Offline/performance/run_1
	@cd $(INT4_DIR) && ./int4_offline -b 512 -a autoconfig_t4 --mlperf_conf_path $(PROJECT_ROOT)/measurements/T4x20/resnet/Offline/mlperf.conf --user_conf_path $(PROJECT_ROOT)/measurements/T4x20/resnet/Offline/user.conf --test-mode PerformanceOnly --lgls_logfile_outdir $(LOG_DIR)/T4x20/resnet/Offline/performance/run_1

# Download benchmark models (weights).
.PHONY: download_model
download_model: link_dataset_dir
	@mkdir -p $(MODEL_DIR)/GNMT
	@if [ ! -f $(MODEL_DIR)/GNMT/gnmt.wts ]; then \
		echo "Downloading GNMT model..." \
			&& wget -nv https://zenodo.org/record/2530924/files/gnmt_model.zip -O $(MODEL_DIR)/GNMT/gnmt_model.zip \
			&& unzip -o $(MODEL_DIR)/GNMT/gnmt_model.zip -d $(MODEL_DIR)/GNMT/ \
			&& rm -f $(MODEL_DIR)/GNMT/gnmt_model.zip \
			&& python3 code/gnmt/tensorrt/convertTFWeights.py -m $(MODEL_DIR)/GNMT/ende_gnmt_model_4_layer/translate.ckpt -o $(MODEL_DIR)/GNMT/gnmt \
			&& rm -rf $(MODEL_DIR)/GNMT/ende_gnmt_model_4_layer; \
	fi

# Generate TensorRT engines (plan files) and run the harness.
.PHONY: run
run:
	@$(MAKE) -f $(MAKEFILE_NAME) generate_engines
	@$(MAKE) -f $(MAKEFILE_NAME) run_harness

# Generate TensorRT engines (plan files).
.PHONY: generate_engines
generate_engines: download_model
	@python3 code/gnmt/tensorrt/main.py $(RUN_ARGS) --action="generate_engines"

# Run the harness and check accuracy if in AccuracyOnly mode.
.PHONY: run_harness
run_harness: download_model
	@python3 code/gnmt/tensorrt/main.py $(RUN_ARGS) --action="run_harness"
	@python3 scripts/print_harness_result.py $(RUN_ARGS)

# Re-generate TensorRT calibration cache.
.PHONY: calibrate
calibrate: download_model
	@python3 code/gnmt/tensorrt/main.py $(RUN_ARGS) --action="calibrate"

############################## UTILITY ##############################
.PHONY: clean
clean: clean_shallow
	rm -rf build

# For clean build.
.PHONY: clean_shallow
clean_shallow:
	rm -rf $(INFERENCE_DIR)/out
	rm -rf $(INFERENCE_DIR)/third_party/gn/out
	rm -f $(INFERENCE_DIR)/third_party/gn/gn*
	rm -rf $(INFERENCE_DIR)/third_party/ninja/build
	rm -f $(INFERENCE_DIR)/third_party/ninja/build.ninja
	rm -f $(INFERENCE_DIR)/third_party/ninja/ninja
ifeq ($(ARCH), x86_64)
	@echo "Cleaning INT4 harness..."
	cd $(INT4_DIR) \
		&& make -j CUDA=$(CUDA_PATH) BUILD_DIR=$(INT4_DIR) LOADGEN_PATH=$(LOADGEN_INCLUDE_DIR) clean
endif

.PHONY: info
info:
	@echo "Architecture=$(ARCH)"
	@echo "User=$(UNAME)"
	@echo "UID=$(UID)"
	@echo "Usergroup=$(GROUPNAME)"
	@echo "GroupID=$(GROUPID)"
	@echo "Docker info: {DETACH=$(DOCKER_DETACH), TAG=$(DOCKER_TAG)}"
	@echo "PATH=$(PATH)"
	@echo "CPATH=$(CPATH)"
	@echo "CUDA_PATH=$(CUDA_PATH)"
	@echo "LIBRARY_PATH=$(LIBRARY_PATH)"
	@echo "LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)"

# The shell target will start a shell that inherits all the environment
# variables set by this Makefile for convenience.
.PHONY: shell
shell:
	@$(SHELL)
