# Instructions for using OpenVino for MLPerf
-----------------------------------
## Requirements
+ OS: Ubuntu (Tested on 18.04 only).
+ GCC (7.4)
+ cmake (Tested with 3.10.2)
+ Python (Tested with 3.6)
+ [OpenCV 4.1.2](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)

## Build MLPerf Loadgen library from mlperf 

(commit: 413dbabcb30dc2ee1fe42e7b8090b37e8144617d  + PR 482 + PR 502)

```
git clone --recurse-submodules https://github.com/mlperf/inference.git mlperf_inference
git pull origin pull/482/head
git pull origin pull/502/head
```

### Build loadgen library

Follow instructions from https://github.com/mlperf/inference/blob/master/loadgen/README_BUILD.md


## Download and build OpenVino

Clone and build OpenVino (opensource dldt) with OpenMP: 
**NB**: In the cmake command sub ```/path/to/opencv/build``` with where you built OpenCV

```
git clone https://github.com/opencv/dldt.git

cd dldt

git submodule update --init --recursive

cd inference-engine

mkdir build && cd build

cmake -DCMAKE_BUILD_TYPE=Release -DTHREADING=OMP -DENABLE_DLIA=OFF -DENABLE_VPU=OFF \
	-DENABPP=OFF -DENABLE_PROFILING_ITT=OFF -DENABLE_VALIDATION_SET=OFF -DENABLE_TESTS=OFF \
	-DENABLE_GNA=OFF -DENABLE_CLDNN=OFF -DENABLE_OPENCV=OFF -DOpenCV_DIR=$( dirname '/path/to/opencv/build/OpenCVConfig.cmake' ) ..

make -j$(nproc)

cd ../bin/intel64/Release/
```

Edit Paths below and export:

```
export LD_LIBRARY_PATH=</path/to/dldt>/inference-engine/temp/omp/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:</path/to/opencv>/build/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:</path/to/dldt>/inference-engine/bin/intel64/Release/lib
```

###  Additional documentations for reference

https://github.com/opencv/dldt/tree/2019/inference-engine (linux)
https://github.com/opencv/dldt/tree/2019/inference-engine#build-on-linux-systems

##  Build ov_mlperf application

mkdir build && cd build

cmake ..

make 


## For Using Quantized SSD-mobilenet model

Please refer to https://github.com/opencv/dldt/blob/pre-release/INT8_WORKFLOW.md

For the MLPerf 0.5 submission, the only directly converted quantized model is ssd-mobilenet from Habana ("ssd-mobilenet 300x300 symmetrically quantized finetuned"), referenced at https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection.


```
sudo bash </path/to/dldt>/model_optimizer/install_prerequisites/install_prerequisites.sh

python3  </path/to/dldt>/model_optimizer/mo.py \
--input_model </path/to/model>/ssd_mobilenet_v1_quant_ft_no_zero_point_frozen_inference_graph.pb \
--input_shape [1,300,300,3] \
--reverse_input_channels \
--tensorflow_use_custom_operations_config </path/to/dldt>/model_optimizer/extensions/front/tf/ssd_v2_support.json \
--tensorflow_object_detection_api_pipeline_config </path/to/model>/pipeline.config
```

##  Running Sample scenarios

Run the corresponding scripts under scripts/ folder
