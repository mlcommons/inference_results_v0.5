cmake_minimum_required(VERSION 3.0)
set(TARGET_NAME ov_mlperf)
project(${TARGET_NAME})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE -fstack-protector-strong -Wno-error")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fno-operator-names -Wformat -Wformat-security -Wall -O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -USE_OPENCV -DBOOST_ERROR_CODE_HEADER_ONLY")
set(CMAKE_BUILD_TYPE "Release")

if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if (NOT(BUILD_DIR))
    MESSAGE(STATUS "BUILD_DIR not defined. Will build to current directory")
    set(BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if (NOT(BIN_FOLDER))
    set(BIN_FOLDER ${BUILD_DIR}/${CMAKE_BUILD_TYPE})
else()
	set(BIN_FOLDER ${BUILD_DIR}/${CMAKE_BUILD_TYPE}_${BIN_FOLDER})
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BIN_FOLDER})

# OpenCV libs
find_package(OpenCV REQUIRED)
find_package(InferenceEngine REQUIRED)

message(STATUS "Adding InferenceEngine_INCLUDE_DIRS:  ${InferenceEngine_INCLUDE_DIRS} ${IE_SRC_DIR}/extension")


# Loadgen
if (NOT(LOADGEN_DIR))
   MESSAGE(FATAL_ERROR "LOADGEN_DIR not passed. Use -DLOADGEN_DIR /path/to/loadgen/")
endif()

if (NOT(LOADGEN_LIB_DIR))
    set(LOADGEN_LIB_DIR ${LOADGEN_DIR})
    message(STATUS "LOADGEN_LIB_DIR unspecified. Set to ${LOADGEN_LIB_DIR}.")
    find_library(LOADGEN_LIBRARY NAMES mlperf_loadgen PATHS "${LOADGEN_DIR}/")
    if (LOADGEN_LIBRARY_NOTFOUND)
        MESSAGE(FATAL_ERROR "Loadgen library not found. Specify parent dir with -DLOADGEN_LIB_DIR")
    endif()
    message(STATUS "LOADGEN_LIBRARY Found: ${LOADGEN_LIBRARY}" )
    
endif()

# Add BOOST
if (NOT(BOOST_SYSTEM_LIB))
    message(FATAL_ERROR "BOOST_SYSTEM_LIB required. Please set -DBOOST_SYSTEM_LIB=/path/to/libboost_system.so")
endif()

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lib)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lib)

#set(SOURCE_FILES backend_ov.h dataset_ov.h sut_ov.h infer_request_wrap.h main_ov.cc)
#add_executable(${TARGET_NAME} ${SOURCE_FILES})

message(STATUS "Loadgen_dir=${LOADGEN_DIR}")
#include_directories("/home/t/MLPerf/mlperf-tracking/inference/loadgen/")
include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}/backend
    ${OpenCV_INCLUDE_DIRS}
    ${LOADGEN_DIR}
    ${InferenceEngine_INCLUDE_DIRS}
    ${IE_SRC_DIR}/extension
    ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/gflags/include
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

#========================== Target =============================================
set(SOURCE_FILES backend_ov.h dataset_ov.h sut_ov.h infer_request_wrap.h item_ov.h main_ov.cc)
add_executable(${TARGET_NAME} ${SOURCE_FILES})
message(STATUS " === InferenceEngine_LIBRARIES: ${InferenceEngine_LIBRARIES} === ")
target_link_libraries(${TARGET_NAME}
        pthread
        ${LOADGEN_LIBRARY}
        ${BOOST_SYSTEM_LIB}
        ${OpenCV_LIBS}
	${IE_RELEASE_LIBRARY}
	${InferenceEngine_LIBRARIES}
        IE::ie_cpu_extension
        ${IE_LIBRARY}
	#"${IE_ROOT_DIR}/external/mkltiny_lnx/lib/libmkl_tiny_omp.so"
    )

