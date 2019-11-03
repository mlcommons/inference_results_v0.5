Instructions for using OpenVino for MLPerf
-----------------------------------
# OS
We tested it on Ubuntu 18.04 only. 

# Prepare

·        Python ,

·        GCC 

·        install OpenCV (instructions : https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)




# Download and build OpenVino

 Building OpenVino with OpenMP: 

 git clone https://github.com/opencv/dldt/tree/pre-release

 cd ./dldt

 git submodule update --init --recursive

 cd ./inference-engine
 
 mkdir build && cd build
 
 cmake -DCMAKE_BUILD_TYPE=Release -DTHREADING=OMP -DENABLE_DLIA=OFF -DENABLE_VPU=OFF -DENABPP=OFF -DENABLE_PROFILING_ITT=OFF -DENABLE_VALIDATION_SET=OFF -DENABLE_TESTS=OFF -DENABLE_GNA=OFF -DENABLE_CLDNN=OFF ..

 make MKLDNNPlugin -j 224
 
 cd ../bin/intel64/Release/

 Please note the library paths in /bin/intel64/Release/

 Edit Paths below and export:

 export InferenceEngine_DIR=/path/to/dldt/inference-engine/build

 export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/openvino/deployment_tools/inference_engine/external/omp/lib


 # Build MLPerf Loadgen cpp library

 git clone --recurse-submodules https://github.com/mlperf/inference.git mlperf_inference

 build loadgen as a C++ library using instructions from https://github.com/mlperf/inference/blob/master/loadgen/README_BUILD.md

 (commit: 413dbabcb30dc2ee1fe42e7b8090b37e8144617d)

# For Using Quantized SSD-mobilenet model

Please refer to https://github.com/opencv/dldt/blob/pre-release/INT8_WORKFLOW.md

For the MLPerf 0.5 submission, the only directly converted quantized model is ssd-mobilenet from Habana ("ssd-mobilenet 300x300 symmetrically quantized finetuned"), referenced at https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection.

$ python3  <OPENVINO_INSTALL_DIR/deployment_tools/model_optimizer/>mo.py
--input_model <path_to_model/>ssd_mobilenet_v1_quant_ft_no_zero_point_frozen_inference_graph.pb
--input_shape [1,300,300,3]
--reverse_input_channels
--tensorflow_use_custom_operations_config <OPENVINO_INSTALL_DIR/deployment_tools/model_optimizer/>extensions/front/tf/ssd_v2_support.json
--tensorflow_object_detection_api_pipeline_config <path_to_model/>pipeline.config


#  Additional documentations for reference

https://github.com/opencv/dldt/blob/pre-release/inference-engine/README.md (linux)
https://github.com/opencv/dldt/blob/pre-release/get-started-linux.md

#  Build ov_mlperf application

mkdir build && cd build

cmake ..

make 

#  Running SingleStream/offline/Server scenarios

Run the corresponding scripts under scripts/ folder