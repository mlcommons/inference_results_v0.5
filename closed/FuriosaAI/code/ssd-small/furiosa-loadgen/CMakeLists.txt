cmake_minimum_required(VERSION 3.0)

project(FuriosaMLPerfRunner)

# versino of furiosa mlpref runner.
set(mlperf_loadgen_VERSION_MAJOR 0)
set(mlperf_loadgen_VERSION_MINOR 1)
message("furiosa_mlperf v${furiosa_mlperf_VERSION_MAJOR}.${furiosa_mlperf_VERSION_MINOR}")

# Set build options.
set(CMAKE_CXX_FLAGS "-O3 -g -march=native -W -Wall -pthread -Wl,-rpath,.")
set(CMAKE_CXX_STANDARD_REQUIRED "ON")
set(CMAKE_CXX_STANDARD "17")
message(STATUS "Using compiler flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Using c++ standard required: ${CMAKE_CXX_STANDARD_REQUIRED}")
message(STATUS "Using c++ standard: ${CMAKE_CXX_STANDAR}")
message(STATUS "Using static linker flags: ${CMAKE_STATIC_LINKER_FLAGS}")
message(STATUS "Using shared linker flags: ${CMAKE_SHARED_LINKER_FLAGS}")

# Add source files.
set(SOURCE
${CMAKE_CURRENT_SOURCE_DIR}/ssd_mobilenet_v1_singlestream.cc
${CMAKE_CURRENT_SOURCE_DIR}/ssd_mobilenet_v1_offline.cc
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../build/include
${CMAKE_CURRENT_SOURCE_DIR}/../dealgood/tensorflow
${CMAKE_CURRENT_SOURCE_DIR}/../dealgood/external/include
${CMAKE_BINARY_DIR}/../include
)
find_library(MLPERF_LOADGEN_LIB libmlperf_loadgen.a PATHS ${CMAKE_BINARY_DIR}/../lib)
find_library(DEALGOOD_LIB libdg.so PATHS ${CMAKE_BINARY_DIR}/../bin)
find_library(TFLITE_LIB libtensorflowlite.so PATHS ${CMAKE_BINARY_DIR}/../bin)
find_library(NUX_LIB libnux.so PATHS ${CMAKE_BINARY_DIR}/../bin)


add_executable(ssd_mobilenet_v1_singlestream ssd_mobilenet_v1_singlestream.cc)
add_executable(ssd_mobilenet_v1_offline ssd_mobilenet_v1_offline.cc)
target_link_libraries(ssd_mobilenet_v1_singlestream ${MLPERF_LOADGEN_LIB} ${DEALGOOD_LIB} ${TFLITE_LIB})
target_link_libraries(ssd_mobilenet_v1_offline ${MLPERF_LOADGEN_LIB} ${DEALGOOD_LIB} ${TFLITE_LIB} tbb)

SET_TARGET_PROPERTIES (ssd_mobilenet_v1_singlestream PROPERTIES INSTALL_RPATH "${CMAKE_BINARY_DIR}/../bin")
SET_TARGET_PROPERTIES (ssd_mobilenet_v1_offline PROPERTIES INSTALL_RPATH "${CMAKE_BINARY_DIR}/../bin")

# Install library and headers.
install(TARGETS ssd_mobilenet_v1_singlestream
        DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
install(TARGETS ssd_mobilenet_v1_offline
        DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
