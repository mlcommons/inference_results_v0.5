# Copyright (c) 2017 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.9)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# add for degug: -lasan -fsanitize=address -fno-omit-frame-pointer -fno-common
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")

project(SPHMLPerf)

set (LOAD_TARGET_NAME "loadrun")

set(LOADGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/loadgen")

set(RUNNER_LIB_NAME "runner_lib")
set(RUNNER_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/runner_lib")
add_subdirectory(${RUNNER_LIB_DIR})

set(NNPI_DIR "/opt/intel_nnpi")

find_package(OpenCV REQUIRED)

set(LINK_LIBRARIES nnpi_inference runner_lib mlperf_loadgen rt ${OpenCV_LIBS})

file (GLOB SRC
	${CMAKE_CURRENT_SOURCE_DIR}/src/*
	)
file (GLOB HEADERS
	${CMAKE_CURRENT_SOURCE_DIR}/*.h
	${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
	)

source_group("src" FILES ${SRC})
source_group("include" FILES ${HEADERS})

link_directories("${LOADGEN_DIR}"  "${NNPI_DIR}/lib")
include_directories("${LOADGEN_DIR}"  "${NNPI_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_executable( ${LOAD_TARGET_NAME} loadrun.cpp  ${SRC} ${HEADERS})

if(UNIX)
	set(LINK_LIBRARIES ${LINK_LIBRARIES} dl pthread)
endif()

target_link_libraries( ${LOAD_TARGET_NAME}  ${LINK_LIBRARIES})
