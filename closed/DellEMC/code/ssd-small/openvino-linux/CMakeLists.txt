cmake_minimum_required(VERSION 3.0)
project(ov_mlperf)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE -fstack-protector-strong")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fno-operator-names -Wformat -Wformat-security -Wall -O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Werror -pthread -USE_OPENCV")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -USE_OPENCV")

#set(CMAKE_EXE_LINKER_FLAGS -nostartfiles -USE_OPENCV)

#set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS}")

set(SOURCE_FILES backend_ov.h dataset_ov.h sut_ov.h infer_request_wrap.h item_ov.h main_ov.cc)
add_executable(ov_mlperf ${SOURCE_FILES})

# OpenCV libs
find_package(OpenCV REQUIRED)
find_package(InferenceEngine 2.0)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(ov_mlperf ${OpenCV_LIBS})


# ==================== PATHS TO SPECIFY! ==================== #

# Eigen lib headers
#include_directories("/usr/local/include/eigen3")

# Protobuf headers


# TensorFlow headers

# Loadgen headers
include_directories("/home/rpremsee/mlperf_inference/mlperf_inference/loadgen")

# Boost headers
#include_directories("/home/rpremsee/mlperf_inference/boost_1_66_install/include/boost")

# OV header
include_directories("/home/rpremsee/mlperf_inference/openvino/dldt/inference-engine/include")
include_directories("/home/rpremsee/mlperf_inference/openvino/dldt/inference-engine/src/extension")

# Link LoadGen libs
target_link_libraries(ov_mlperf "/home/rpremsee/mlperf_inference/mlperf_inference/loadgen/libmlperf_loadgen.a")

# Link Boost libs
#target_link_libraries(ov_mlperf "/home/rpremsee/mlperf_inference/boost_1_66_install/lib/libboost_filesystem.so")
#target_link_libraries(ov_mlperf "/home/rpremsee/mlperf_inference/boost_1_66_install/lib/libboost_system.so")

#target_link_libraries(ov_mlperf "/usr/local/lib/libprotobuf-lite.so")

# Link to OV libs
target_link_libraries(ov_mlperf "/home/rpremsee/mlperf_inference/openvino/dldt/inference-engine/bin/intel64/Release/lib/libcpu_extension.so")
target_link_libraries(ov_mlperf "/home/rpremsee/mlperf_inference/openvino/dldt/inference-engine/bin/intel64/Release/lib/libinference_engine.so")
target_link_libraries(ov_mlperf "/home/rpremsee/mlperf_inference/openvino/dldt/inference-engine/temp/mkltiny_lnx_20190620/lib/libmkl_tiny_omp.so")

#target_link_libraries(ov_mlperf "/home/intel/mlperf_inference/openvino/ov_install/openvino/inference_engine/lib/intel64/libMKLDNNPlugin.so")

