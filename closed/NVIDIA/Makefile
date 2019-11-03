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
ifneq ($(wildcard /home/scratch.mlperf_inference/preprocessed_data),)
	MLPERF_INFERENCE_PATH := /home/scratch.mlperf_inference
endif
# Toto
ifneq ($(wildcard /scratch/datasets/mlperf_inference/preprocessed_data),)
	MLPERF_INFERENCE_PATH := /scratch/datasets/mlperf_inference
endif
# Circe
ifneq ($(wildcard /gpfs/fs1/datasets/mlperf_inference/preprocessed_data),)
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
# Build the docker image and launch an interactive container.
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

# Build the docker image.
.PHONY: build_docker
build_docker: link_dataset_dir
ifeq ($(ARCH), x86_64)
	@echo "Building Docker image"
	docker build -t mlperf-inference:$(DOCKER_TAG)-latest \
		--network host -f docker/Dockerfile .
endif

# Add current user into docker image.
.PHONY: docker_add_user
docker_add_user:
ifeq ($(ARCH), x86_64)
	@echo "Adding user account into image"
	docker build -t mlperf-inference:$(DOCKER_TAG) --network host \
		--build-arg BASE_IMAGE=mlperf-inference:$(DOCKER_TAG)-latest \
		--build-arg GID=$(GROUPID) --build-arg UID=$(UID) --build-arg GROUP=$(GROUPNAME) --build-arg USER=$(UNAME) \
		- < docker/Dockerfile.user
endif

# Launch an interactive container.
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

# Download COCO datasets and GNMT inference data. Imagenet does not have public links.
.PHONY: download_dataset
download_dataset:
	@mkdir -p $(DATA_DIR)/imagenet
	@mkdir -p $(DATA_DIR)/coco
	@wget http://images.cocodataset.org/zips/train2017.zip -O $(DATA_DIR)/coco/train2017.zip
	@wget http://images.cocodataset.org/zips/val2017.zip -O $(DATA_DIR)/coco/val2017.zip
	@wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O $(DATA_DIR)/coco/annotations_trainval2017.zip
	@cd $(DATA_DIR)/coco && unzip train2017.zip
	@cd $(DATA_DIR)/coco && unzip val2017.zip
	@cd $(DATA_DIR)/coco && unzip annotations_trainval2017.zip
	@mkdir -p $(DATA_DIR)/nmt/GNMT
	@wget -nv https://zenodo.org/record/3437893/files/gnmt_inference_data.zip -O $(DATA_DIR)/nmt/GNMT/gnmt_data.zip
	@unzip -j -o $(DATA_DIR)/nmt/GNMT/gnmt_data.zip -d $(DATA_DIR)/nmt/GNMT/

# Preprocess the data. See scripts/preprocess_data.py for more information.
.PHONY: preprocess_data
preprocess_data:
	@python3 scripts/preprocess_data.py --data_dir="$$DATA_DIR" --output_dir="$$PREPROCESSED_DATA_DIR"
	@cp -RT $(DATA_DIR)/nmt $(PREPROCESSED_DATA_DIR)/nmt

# Download benchmark models (weights).
.PHONY: download_model
download_model: link_dataset_dir
	@mkdir -p $(MODEL_DIR)/ResNet50
	@if [ ! -f $(MODEL_DIR)/ResNet50/resnet50_v1.onnx ]; then \
		echo "Downloading ResNet50 model..." \
			&& wget -nv https://zenodo.org/record/2592612/files/resnet50_v1.onnx -O $(MODEL_DIR)/ResNet50/resnet50_v1.onnx ; \
	fi
	@mkdir -p $(MODEL_DIR)/MobileNet
	@if [ ! -f $(MODEL_DIR)/MobileNet/mobilenet_sym_no_bn.onnx ]; then \
		echo "Downloading MobileNet model..." \
			&& wget -nv https://zenodo.org/record/3353417/files/Quantized%20MobileNet.zip -O $(MODEL_DIR)/MobileNet/MobileNet.zip \
			&& unzip -j $(MODEL_DIR)/MobileNet/MobileNet.zip Quantized\ MobileNet/mobilenet_sym_no_bn.onnx -d $(MODEL_DIR)/MobileNet/ \
			&& rm -f $(MODEL_DIR)/MobileNet/MobileNet.zip; \
	fi
	@mkdir -p $(MODEL_DIR)/SSDResNet34
	@if [ ! -f $(MODEL_DIR)/SSDResNet34/resnet34-ssd1200.pytorch ]; then \
		echo "Downloading SSDResNet34 model..." \
			&& wget -nv https://zenodo.org/record/3236545/files/resnet34-ssd1200.pytorch -O $(MODEL_DIR)/SSDResNet34/resnet34-ssd1200.pytorch; \
	fi
	@mkdir -p $(MODEL_DIR)/SSDMobileNet
	@if [ ! -f $(MODEL_DIR)/SSDMobileNet/frozen_inference_graph.pb ]; then \
		echo "Downloading SSDMobileNet model..." \
			&& wget -nv http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz -O $(MODEL_DIR)/SSDMobileNet/SSDMobileNet.tar.gz \
			&& tar -xzvf $(MODEL_DIR)/SSDMobileNet/SSDMobileNet.tar.gz -C $(MODEL_DIR)/SSDMobileNet/ --strip-components 1 ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb \
			&& rm -f $(MODEL_DIR)/SSDMobileNet/SSDMobileNet.tar.gz; \
	fi
	@mkdir -p $(MODEL_DIR)/GNMT
	@if [ ! -f $(MODEL_DIR)/GNMT/gnmt.wts ]; then \
		echo "Downloading GNMT model..." \
			&& wget -nv https://zenodo.org/record/2530924/files/gnmt_model.zip -O $(MODEL_DIR)/GNMT/gnmt_model.zip \
			&& unzip -o $(MODEL_DIR)/GNMT/gnmt_model.zip -d $(MODEL_DIR)/GNMT/ \
			&& rm -f $(MODEL_DIR)/GNMT/gnmt_model.zip \
			&& python3 code/gnmt/tensorrt/convertTFWeights.py -m $(MODEL_DIR)/GNMT/ende_gnmt_model_4_layer/translate.ckpt -o $(MODEL_DIR)/GNMT/gnmt \
			&& rm -rf $(MODEL_DIR)/GNMT/ende_gnmt_model_4_layer; \
	fi

############################### BUILD ###############################

# Build all source codes.
.PHONY: build
build: clone_loadgen link_dataset_dir
	@$(MAKE) -f $(MAKEFILE_NAME) build_gnmt
	@$(MAKE) -f $(MAKEFILE_NAME) build_plugins
	@$(MAKE) -f $(MAKEFILE_NAME) build_loadgen
	@$(MAKE) -f $(MAKEFILE_NAME) build_harness

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

# Build GNMT source codes.
.PHONY: build_gnmt
build_gnmt:
ifeq ($(ARCH), x86_64)
	@echo "Building GNMT source..."
	cd code/gnmt/tensorrt/src \
		&& make -j
endif

# Build TensorRT plugins.
.PHONY: build_plugins
build_plugins:
	mkdir -p build/plugins/NMSOptPlugin
	cd build/plugins/NMSOptPlugin \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(PROJECT_ROOT)/code/plugin/NMSOptPlugin \
		&& make -j

# Build LoadGen.
.PHONY: build_loadgen
build_loadgen:
	@echo "Building loadgen..."
	@cd $(INFERENCE_DIR)/third_party/ninja \
		&& python2 configure.py --bootstrap \
		&& cd ../.. \
		&& python2 third_party/gn/build/gen.py \
		&& third_party/ninja/ninja -C third_party/gn/out \
		&& cp third_party/gn/out/gn* third_party/gn/. \
		&& third_party/gn/gn gen out/MakefileGnProj --args="is_debug=$(LOADGEN_DEBUG_BUILD)" \
		&& third_party/ninja/ninja -C out/MakefileGnProj mlperf_loadgen

# Build harness source codes.
.PHONY: build_harness
build_harness:
	@echo "Building harness..."
	@mkdir -p build/harness \
		&& cd build/harness \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DLOADGEN_INCLUDE_DIR=$(LOADGEN_INCLUDE_DIR) $(PROJECT_ROOT)/code/harness \
		&& make -j
	@echo "Finished building harness."

###############################  RUN  ###############################

# Generate TensorRT engines (plan files) and run the harness.
.PHONY: run
run:
	@$(MAKE) -f $(MAKEFILE_NAME) generate_engines
	@$(MAKE) -f $(MAKEFILE_NAME) run_harness

# Generate TensorRT engines (plan files).
.PHONY: generate_engines
generate_engines: download_model
	@python3 code/main.py $(RUN_ARGS) --action="generate_engines"

# Run the harness and check accuracy if in AccuracyOnly mode.
.PHONY: run_harness
run_harness: download_model
	@python3 code/main.py $(RUN_ARGS) --action="run_harness"
	@python3 scripts/print_harness_result.py $(RUN_ARGS)

# Re-generate TensorRT calibration cache.
.PHONY: calibrate
calibrate: download_model
	@python3 code/main.py $(RUN_ARGS) --action="calibrate"

############################## UTILITY ##############################

# Move LoadGen logs from build/logs to results/.
.PHONY: update_results
update_results:
	@python3 scripts/update_results.py

# Remove build directory.
.PHONY: clean
clean: clean_shallow
	rm -rf build

# Remove only the files necessary for a clean build.
.PHONY: clean_shallow
clean_shallow:
	rm -rf build/bin
	rm -rf build/harness
	rm -rf build/plugins
	rm -rf $(INFERENCE_DIR)/out
	rm -rf $(INFERENCE_DIR)/third_party/gn/out
	rm -f $(INFERENCE_DIR)/third_party/gn/gn*
	rm -rf $(INFERENCE_DIR)/third_party/ninja/build
	rm -f $(INFERENCE_DIR)/third_party/ninja/build.ninja
	rm -f $(INFERENCE_DIR)/third_party/ninja/ninja

# Print out useful information.
.PHONY: info
info:
	@echo "Architecture=$(ARCH)"
	@echo "User=$(UNAME)"
	@echo "UID=$(UID)"
	@echo "Usergroup=$(GROUPNAME)"
	@echo "GroupID=$(GROUPID)"
	@echo "Docker info: {DETACH=$(DOCKER_DETACH), TAG=$(DOCKER_TAG)}"
	@echo "TensorRT Version=$(TRT_VER)"
	@echo "CUDA Version=$(CUDA_VER)"
	@echo "CuDNN Version=$(CUDNN_VER)"
	@echo "Ubuntu Version=$(UBUNTU_VER)"
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
