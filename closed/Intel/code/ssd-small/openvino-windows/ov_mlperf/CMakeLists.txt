cmake_minimum_required(VERSION 3.0)
set(TARGET_NAME ov_mlperf)
project(${TARGET_NAME})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE -fstack-protector-strong") # -Wno-error")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fno-operator-names") # -Wformat -Wformat-security -Wall -O2")
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

message(STATUS "Loadgen_dir=${LOADGEN_DIR}")

include_directories (
    ${OpenCV_INCLUDE_DIRS}
    ${LOADGEN_DIR}
	${BOOST_INCLUDE_DIRS}
    ${InferenceEngine_INCLUDE_DIRS}
    ${IE_SRC_DIR}/extension
)

set(SOURCE_FILES backend_ov.h dataset_ov.h sut_ov.h infer_request_wrap.h main_ov.cc)
add_executable(${TARGET_NAME} ${SOURCE_FILES})

target_link_libraries(${TARGET_NAME}
        ${LOADGEN_LIBRARY}
        ${OpenCV_LIBS}
        ${IE_EXTENSION_LIB}
        ${IE_LIBRARY}
    )
