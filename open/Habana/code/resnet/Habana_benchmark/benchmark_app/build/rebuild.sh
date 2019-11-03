#!/bin/bash
CONFIG="$1"
[ "$CONFIG" == "" ] && { echo "usage: ./build.sh Debug/Release"; exit 1; }
echo "building: " $CONFIG
if [ -d "$CONFIG" ]; then
    echo $CONFIG " exists, clearing content"
    cd $CONFIG
    rm * -r
else
    echo "make directory: " $CONFIG
    mkdir $CONFIG
    cd $CONFIG
fi
if [ -d "$MLPREF_BENCHMARK_CODE_DIR" ]; then
    cmake -DCMAKE_BUILD_TYPE=$1 -DENABLE_SANITIZER=OFF -DCMAKE_CXX_COMPILER=clang++-8  $MLPREF_BENCHMARK_CODE_DIR
    make -B
else
    exit 1
fi