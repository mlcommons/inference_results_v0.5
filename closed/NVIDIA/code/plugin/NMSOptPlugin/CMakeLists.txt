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

# We need cmake >= 3.8, since 3.8 introduced CUDA as a first class language
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(NMSOptPluginSSD LANGUAGES CXX CUDA)

# Enable all compile warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wno-deprecated-declarations")

# -------- CONFIGURATION --------
if (NOT $ENV{TRT_LIB_DIR})
  set(TRT_LIB_DIR /usr/lib/x86_64-linux-gnu)
else()
  set(TRT_LIB_DIR $ENV{TRT_LIB_DIR})
endif()
if (NOT $ENV{TRT_INCLUDE_DIR})
  set(TRT_INCLUDE_DIR /usr/include/x86_64-linux-gnu)
else()
  set(TRT_INCLUDE_DIR $ENV{TRT_INCLUDE_DIR})
endif()
if (NOT $ENV{CUB_DIR})
  set(CUB_DIR /usr/local/cub)
else()
  set(CUB_DIR $ENV{CUB_DIR})
endif()
if (NOT $ENV{CUDNN_LIB_DIR})
  set(CUDNN_LIB_DIR /usr/lib/x86_64-linux-gnu)
else()
  set(CUDNN_LIB_DIR $ENV{CUDNN_LIB_DIR})
endif()
if (NOT $ENV{CUDNN_INC_DIR})
  set(CUDNN_INC_DIR /usr/include/x86_64-linux-gnu)
else()
  set(CUDNN_INC_DIR $ENV{CUDNN_INC_DIR})
endif()
if (NOT $ENV{CUDA_INC_DIR})
  set(CUDA_INC_DIR /usr/local/cuda/include)
else()
  set(CUDA_INC_DIR $ENV{CUDA_INC_DIR})
endif()

# Find dependencies:
message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")

# TensorRT's nvinfer lib
string(REPLACE ":" ";" LIBRARY_DIRS $ENV{LD_LIBRARY_PATH})
find_library(NVINFER_LIB nvinfer HINTS ${TRT_LIB_DIR} ${LIBRARY_DIRS} PATH_SUFFIXES lib lib64)
find_library(CUDNN_LIB cudnn HINTS ${CUDNN_LIB_DIR} ${LIBRARY_DIRS} PATH_SUFFIXES lib lib64)

# -------- BUILDING --------

# Add include directories
include_directories(${CUDA_INC_DIR} ${TRT_INCLUDE_DIR} ${CUB_DIR} ${CUDNN_INC_DIR} ${CMAKE_SOURCE_DIR}/src/)

# Define nms opt plugin library target
FILE(GLOB SOURCE_CU_FILES ${CMAKE_SOURCE_DIR}/src/*.cu)
FILE(GLOB SOURCE_CPP_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)
FILE(GLOB HEADER_FILES ${CMAKE_SOURCE_DIR}/src/*.h)
add_library(nmsoptplugin MODULE
  ${SOURCE_CU_FILES}
  ${SOURCE_CPP_FILES}
  ${HEADER_FILES}
)

# Use C++11
target_compile_features(nmsoptplugin PUBLIC cxx_std_11)

# Set sm versions
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_70,code=sm_70")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_72,code=sm_72")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_75,code=sm_75")

# Link TensorRT's nvinfer lib
target_link_libraries(nmsoptplugin PUBLIC ${NVINFER_LIB})
target_link_libraries(nmsoptplugin PUBLIC ${CUDNN_LIB})

# We need to explicitly state that we need all CUDA files
# to be built with -dc as the member functions will be called by
# other libraries and executables (in our case, Python inference scripts)
set_target_properties(nmsoptplugin PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
