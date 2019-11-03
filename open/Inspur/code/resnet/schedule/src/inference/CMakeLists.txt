cmake_minimum_required(VERSION 3.9)

project(inference C CXX CUDA)

set(CMAKE_CXX_STANDARD "17")
message("dir" ${CMAKE_CURRENT_LIST_DIR})
file(GLOB INFERENCE_SRC
	${CMAKE_CURRENT_LIST_DIR}/src/schedule/*.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/schedule/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/inference/*.cc
	${CMAKE_CURRENT_SOURCE_DIR}/src/inference/*.cu
	${CMAKE_CURRENT_SOURCE_DIR}/src/inference/*.h
     )

set(TENSORRT_INCLUDE_DIRS /media/sunhy/inference/TensorRT-5.1/include/)
set(TENSORRT_INCLUDE_LIBS /media/sunhy/inference/TensorRT-5.1/targets/x86_64-linux-gnu/lib/)
INCLUDE_DIRECTORIES(${TENSORRT_INCLUDE_DIRS})
#target_link_libraries(TENSORRT_INCLUDE_LIBS)
FIND_PACKAGE(PythonInterp 3)
FIND_PACKAGE(PythonLibs 3)

FIND_PATH(PYTHON_INCLUDE_PATH Python.h
  /usr/include
  /usr/local/include)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})

find_package(CUDA 8.0 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

find_path   (CUDNN_INCLUDE NAMES cudnn.h)
find_path   (CUDA_INCLUDE NAMES cudnn.h)
find_package_handle_standard_args(CUDNN DEFAULT_MSG CUDNN_INCLUDE CUDNN_LIBRARY)
set(INFERENCE_COMPILE_CODE ${INFERENCE_SRC})
add_library(inference ${INFERENCE_COMPILE_CODE})
target_link_libraries(inference cublas cudnn cuda)

