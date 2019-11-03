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

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(mlperf-inference)

include(GNUInstallDirs)
find_package(CUDA REQUIRED)

# Build options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

# Pass the Loadgen include directory from command line
add_definitions(-DLOADGEN_INCLUDE_DIR=${LOADGEN_INCLUDE_DIR})

# Workaround for TRT header warning
execute_process(COMMAND echo "Warning: setting -Wno-deprecated-declarations to avoid header warnings")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

project(harness LANGUAGES CXX CUDA)

# Find the static Loadgen library
unset(LOADGEN_LIB CACHE)
find_library(LOADGEN_LIB NAMES libmlperf_loadgen.a PATHS ../../../../build/inference/out/MakefileGnProj/obj/loadgen)

######### GNMT HARNESS ########
set(GNMT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../../../../code/gnmt/tensorrt)

find_library(GNMT_CORE_LIB NAMES libgnmtcore.a PATHS ../../../../build/bin/GNMT)
find_library(SIMPLEJSON_LIB NAMES libSimpleJSON.a PATHS /usr/lib/aarch64-linux-gnu)

execute_process(COMMAND echo "Building offline GNMT harness...")

add_executable(harness_gnmt_default
    ./main_gnmt_offline.cc
    ./gnmt_server.cc
)

# Link to libgnmtcore.a with --whole-archive to make sure plugins are registered.
target_link_libraries(harness_gnmt_default
    -Wl,--whole-archive
    ${GNMT_CORE_LIB}
    -Wl,--no-whole-archive
    nvinfer
    nvinfer_plugin
    gflags
    glog
    pthread
    cublas
    ${CUDA_LIBRARIES}
    ${LOADGEN_LIB}
    ${SIMPLEJSON_LIB}
)

target_include_directories(harness_gnmt_default
    PUBLIC
        common
    ${GNMT_SOURCE_DIR}/src/
    ${GNMT_SOURCE_DIR}/src/core/
    ${GNMT_SOURCE_DIR}/src/plugin/
    ${GNMT_SOURCE_DIR}/src/common/
    ./
    ../../../../build/inference/loadgen
)

target_compile_features(harness_gnmt_default PUBLIC cxx_std_14)
