#!/bin/bash
PWD="$(pwd)"
if [ ! -d "$MLPERF_BASE_PATH" ]; then
  mkdir --parents $MLPERF_BASE_PATH
fi
cd $MLPERF_BASE_PATH
git clone --recurse-submodules https://github.com/mlperf/inference.git 
cd $MLPERF_BASE_PATH/inference/loadgen
git checkout 413dbabcb30dc2ee1fe42e7b8090b37e8144617d
git pull origin pull/482/head
git pull origin pull/489/head
git pull origin pull/502/head

cd ..
make mlperf_loadgen

cd $PWD 