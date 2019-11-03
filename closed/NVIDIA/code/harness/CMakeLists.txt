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
find_library(LOADGEN_LIB NAMES libmlperf_loadgen.a PATHS ../../build/inference/out/MakefileGnProj/obj/loadgen)

# Set the path to the LWIS library
unset(LWIS_INCLUDE_DIR CACHE)
set(LWIS_INCLUDE_DIR lwis/include)

# Build the harness for the server scenario and for GNMT if not on "aarch64" platform
execute_process(COMMAND uname -p OUTPUT_VARIABLE ARCH)

######### GNMT HARNESS ########
if (NOT ${ARCH} MATCHES "aarch64")
    set(GNMT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../../code/gnmt/tensorrt)

    find_library(GNMT_CORE_LIB NAMES libgnmtcore.a PATHS ../../build/bin/GNMT)
    find_library(SIMPLEJSON_LIB NAMES libSimpleJSON.a PATHS /usr/lib/x86_64-linux-gnu)

    execute_process(COMMAND echo "Building server GNMT harness...")

    add_definitions(-DTRT_LAB)

    add_executable(harness_gnmt_server
        harness_gnmt_server/main_server.cc
        harness_gnmt_server/GNMTModel.cc
        harness_gnmt_server/GNMTBindings.cc
        harness_gnmt_server/GNMTQSL.cc
        harness_gnmt_server/GNMTSUT.cc
        harness_gnmt_server/GNMTBench.cc
        harness_gnmt_server/GNMTBindingBase.cc
        harness_gnmt_server/common.cc
    )

    # Link to libgnmtcore.a with --whole-archive to make sure plugins are registered.
    target_link_libraries(harness_gnmt_server
        -Wl,--whole-archive
        ${GNMT_CORE_LIB}
        -Wl,--no-whole-archive
        trtlab
        trtlab-core
        trtlab-cuda
        nvidia-ml
        nvinfer
        nvinfer_plugin
        gflags
        glog
        pthread
        cublasLt
        cublas
        ${CUDA_LIBRARIES}
        ${LOADGEN_LIB}
        ${SIMPLEJSON_LIB}
    )

    target_include_directories(harness_gnmt_server
        PUBLIC
            ${CUDA_INCLUDE_DIRS}
            common
        ${GNMT_SOURCE_DIR}/src/
        ${GNMT_SOURCE_DIR}/src/core/
        ${GNMT_SOURCE_DIR}/src/plugin/
        ${GNMT_SOURCE_DIR}/src/common/
        harness_gnmt_server/
        ../../build/inference/loadgen
    )

    target_compile_features(harness_gnmt_server PUBLIC cxx_std_14)

    execute_process(COMMAND echo "Building offline GNMT harness...")

    add_executable(harness_gnmt_default
        harness_gnmt_default/main_gnmt_offline.cc
        harness_gnmt_default/gnmt_server.cc
    )

    # Link to libgnmtcore.a with --whole-archive to make sure plugins are registered.
    target_link_libraries(harness_gnmt_default
        -Wl,--whole-archive
        ${GNMT_CORE_LIB}
        -Wl,--no-whole-archive
        trtlab
        trtlab-core
        trtlab-cuda
        nvidia-ml
        nvinfer
        nvinfer_plugin
        gflags
        glog
        pthread
        cublasLt
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
        harness_gnmt_default/
        ../../build/inference/loadgen
    )

    target_compile_features(harness_gnmt_default PUBLIC cxx_std_14)

else()
    execute_process(COMMAND echo "Skipping GNMT harness for ${ARCH}")
endif()

######### DEFAULT HARNESS ########
# Add the LWIS subdirectory (which will generate a static LWIS library)
add_subdirectory(lwis)

# Build the default harness which covers single_stream, offline, and multi_stream sceanrios on image benchmarks.
execute_process(COMMAND echo "Building default harness...")
add_executable(harness_default
    harness_default/main_default.cc
    common/logger.cpp
)

target_link_libraries(harness_default
    nvinfer
    nvinfer_plugin
    gflags
    glog
    ${CUDA_LIBRARIES}
    lwis
    ${LOADGEN_LIB}
)

target_include_directories(harness_default
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${LOADGEN_INCLUDE_DIR}
        ${LWIS_INCLUDE_DIR}
        common
)
