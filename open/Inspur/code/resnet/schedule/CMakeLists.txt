cmake_minimum_required(VERSION 3.9)

project(schedule C CXX CUDA)

set(CMAKE_CXX_STANDARD "17")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
else()
	set(OpenCV_DIR D:\\work\\libs\\opencv4.1.1\\OpenCV\\build)
endif()

#find_package(Boost COMPONENTS thread REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(TENSORRT_INCLUDE_DIRS /root/tools/TensorRT-5.1/include/)
set(TENSORRT_PLUGIN_DIRS /root/tools/TensorRT-5.1/targets/x86_64-linux-gnu/samples/samplePlugin/)
set(TENSORRT_LIBRARIES /root/tools/TensorRT-5.1/targets/x86_64-linux-gnu/lib/)
#set(TENSORRT_INCLUDE_DIRS /media/sunhy/inference/TensorRT-5.1/include/)
#set(TENSORRT_PLUGIN_DIRS /media/sunhy/inference/TensorRT-5.1/targets/x86_64-linux-gnu/samples/samplePlugin/)
#set(TENSORRT_LIBRARIES /media/sunhy/inference/TensorRT-5.1/targets/x86_64-linux-gnu/lib/)

#message(STATUS "Boost status:")
#MESSAGE(STATUS "    Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}.")
#MESSAGE(STATUS "    Boost_LIBRARIES = ${Boost_LIBRARIES}.")
#MESSAGE(STATUS "    Boost_LIB_VERSION = ${Boost_LIB_VERSION}.")

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

message(STATUS "CUDA status:")
message(STATUS "    cuda libraries: ${CUDA_LIBRARIES}")

#include_directories(${Boost_INCLUDE_DIRS})
include_directories(${ZLIB_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TENSORRT_INCLUDE_DIRS})
include_directories(${TENSORRT_PLUGIN_DIRS})
link_directories("/usr/local/cuda/lib")

message(STATUS "ZLIB_INCLUDE_DIRS: ${ZLIB_INCLUDE_DIRS}")
message(STATUS "ZLIB_LIBRARIES: ${ZLIB_LIBRARIES}")

file(GLOB SCHEDULE_SRC
	${CMAKE_CURRENT_LIST_DIR}/src/*.h
	${CMAKE_CURRENT_LIST_DIR}/src/*.cpp

	${CMAKE_CURRENT_LIST_DIR}/src/common/*.h
	${CMAKE_CURRENT_LIST_DIR}/src/common/*.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/config/*.h
	${CMAKE_CURRENT_LIST_DIR}/src/config/*.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/bindings/*.h
	${CMAKE_CURRENT_LIST_DIR}/src/postprocess/*.h
	${CMAKE_CURRENT_LIST_DIR}/src/postprocess/*.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/inference/*.h
	${CMAKE_CURRENT_LIST_DIR}/src/inference/*.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/inference/*.cu
	${CMAKE_CURRENT_LIST_DIR}/src/inference/*.cc
	${CMAKE_CURRENT_LIST_DIR}/src/schedule/*.h
	${CMAKE_CURRENT_LIST_DIR}/src/schedule/*.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/schedule/settings/*.h
	${CMAKE_CURRENT_LIST_DIR}/src/schedule/settings/*.cpp

	${CMAKE_CURRENT_SOURCE_DIR}/src/loadgen/bindings/c_api.h
	${CMAKE_CURRENT_SOURCE_DIR}/src/loadgen/bindings/c_api.cc
	${CMAKE_CURRENT_SOURCE_DIR}/src/loadgen/loadgen.cc
	${CMAKE_CURRENT_SOURCE_DIR}/src/loadgen/logging.h
	${CMAKE_CURRENT_SOURCE_DIR}/src/loadgen/logging.cc
	${CMAKE_CURRENT_SOURCE_DIR}/src/loadgen/utils.h
	${CMAKE_CURRENT_SOURCE_DIR}/src/loadgen/utils.cc
	${CMAKE_CURRENT_SOURCE_DIR}/src/loadgen/test_settings_internal.h
	${CMAKE_CURRENT_SOURCE_DIR}/src/loadgen/test_settings_internal.cc
	${CMAKE_CURRENT_SOURCE_DIR}/src/loadgen/version.h
	${CMAKE_CURRENT_SOURCE_DIR}/src/loadgen/version.cc
	${CMAKE_CURRENT_SOURCE_DIR}/src/loadgen/build/version_generated.cc

	${CMAKE_CURRENT_SOURCE_DIR}/src/cnpy/cnpy.h
	${CMAKE_CURRENT_SOURCE_DIR}/src/cnpy/cnpy.cpp
	)

set(SCHEDULE_COMPILE_CODE ${SCHEDULE_SRC})
add_executable(mlperf_schedule ${SCHEDULE_COMPILE_CODE})
#add_library(mlperf_schedule STATIC ${SCHEDULE_COMPILE_CODE})
#target_link_libraries(mlperf_schedule)

target_compile_features(mlperf_schedule PUBLIC cxx_std_14)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_SEPARABLE_COMPILATION ON)

# Link your application with OpenCV libraries
#target_link_libraries (mlperf_schedule ${Boost_LIBRARIES})
target_link_libraries(mlperf_schedule ${OpenCV_LIBS})
target_link_libraries(mlperf_schedule ${CUDA_LIBS})
target_link_libraries(mlperf_schedule cudart cudnn cublas)
target_link_libraries(mlperf_schedule ${TENSORRT_LIBRARIES}/libnvinfer.so 
	${TENSORRT_LIBRARIES}/libnvinfer_plugin.so 
	${TENSORRT_LIBRARIES}/libnvparsers.so
	)
target_link_libraries(mlperf_schedule ${ZLIB_LIBRARIES})

#if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
#	target_link_libraries(mlperf_schedule ${CMAKE_SOURCE_DIR}/src/third_party/loadgen/lib/libmlperf_loadgen.a)
#else()
#	target_link_libraries(mlperf_schedule ${CMAKE_SOURCE_DIR}/src/third_party/loadgen/lib/mlperf_loadgen.lib)
#endif()
