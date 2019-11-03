cmake_minimum_required(VERSION 3.9)

project(inference C CXX CUDA)

set(CMAKE_CXX_STANDARD "17")

file(GLOB INFERENCE_SRC
	${CMAKE_CURRENT_LIST_DIR}/src/schedule/*.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/schedule/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/inference/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/inference/*.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/inference/*.h
     )

set(TENSORRT_INCLUDE_DIRS /media/sunhy/inference/TensorRT-5.1/include/)
set(TENSORRT_LIBRARIES /media/sunhy/inference/TensorRT-5.1/targets/x86_64-linux-gnu/lib/)

set(TENSORRT_PLUGIN_DIRS /media/sunhy/inference/TensorRT-5.1/targets/x86_64-linux-gnu/samples/samplePlugin/)
INCLUDE_DIRECTORIES(${TENSORRT_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${TENSORRT_PLUGIN_DIRS})
#target_link_libraries(TENSORRT_INCLUDE_LIBS)
FIND_PACKAGE(PythonInterp 3)
FIND_PACKAGE(PythonLibs 3)

FIND_PATH(PYTHON_INCLUDE_PATH Python.h
  /usr/include
  /usr/local/include)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src/inference/ssd_Calibration)

find_package(CUDA 8.0 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_libraries(${CUDA_LIBS})
message(STATUS "   cuda libraries: ${CUDA_LIBRARIES}")
message(STATUS "   python libraries: ${PythonLibs}")

#find_package_handle_standard_args(CUDNN DEFAULT_MSG CUDNN_INCLUDE CUDNN_LIBRARY)
set(INFERENCE_COMPILE_CODE ${INFERENCE_SRC})
add_library(inference SHARED ${INFERENCE_COMPILE_CODE})

target_link_libraries(inference ${TENSORRT_LIBRARIES}/libnvinfer.so 
${TENSORRT_LIBRARIES}/libnvinfer_plugin.so 
 ${TENSORRT_LIBRARIES}/libnvparsers.so)
target_link_libraries(inference ${PYTHON_LIBRARIES})
target_link_libraries(inference ${CMAKE_CURRENT_SOURCE_DIR}/src/inference/uff_custom_plugin/build/libclipplugin.so)
target_link_libraries(inference ${CUDA_LIBRARIES})
target_link_libraries(inference cublas cudnn cuda)
