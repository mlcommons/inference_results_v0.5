#!/bin/bash
export BASE_PATH=/home/rshefer/projects
export MLPERF_BASE_PATH=$BASE_PATH/MLPERF/
export MLPERF_LOADGEN_PATH=$MLPERF_BASE_PATH/inference/loadgen/
export MLPERF_BUILD_DIR=$BASE_PATH/habana_build_dir/
export MLPERF_MODEL_DIR=$BASE_PATH/mlperf_models/
export MLPERF_LIB_PATH=$MLPERF_BASE_PATH/inference/out/MakefileGnProj/obj/loadgen/
export MLPREF_BENCHMARK_CODE_DIR=$BASE_PATH/demos/Habana_benchmark/
export MLPERF_INSTALL_DIR=$BASE_PATH/habana_benchmark_install/
if [ ! -d "$MLPERF_BASE_PATH" ]; then
  mkdir --parents $MLPERF_BASE_PATH
fi

if [ ! -d "$MLPERF_BUILD_DIR" ]; then
  mkdir --parents  $MLPERF_BUILD_DIR
fi

if [ ! -d "$MLPERF_INSTALL_DIR" ]; then
  mkdir --parents $MLPERF_INSTALL_DIR
fi
