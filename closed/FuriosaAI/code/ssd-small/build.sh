#!/bin/bash
cd "$(dirname "$0")"
mkdir ./build
mkdir ./build/mlperf-loadgen
mkdir ./build/furiosa-loadgen
mkdir ./build/bin
mkdir ./build/lib
mkdir ./build/include

# Build mlperf loadgen
cd ./build/mlperf-loadgen
cmake -DCMAKE_CXX_STANDARD=14 -DCMAKE_CXX_STANDARD_REQUIRED=ON -DCMAKE_INSTALL_PREFIX=../ ../../inference/loadgen
cmake --build . --target install

# Build furiosa mlperf
cd ../../build/furiosa-loadgen
if [[ -f "Makefile" ]]; then
  echo 'reuse Makefile'
else
  cmake -DCMAKE_INSTALL_PREFIX=../ ../../furiosa-loadgen
fi
cmake --build . --target install
