#~/bin/bash

# CAFFE2_DIR=../../

batchsize_i=$1
iteration_i=$2
scenario=$3
target_qps=$4
spec_override=$5


export CAFFE2_INFERENCE_MEM_OPT=1

export OMP_NUM_THREADS=56  KMP_AFFINITY="proclist=[56-111],granularity=fine,explicit"

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CAFFE2_DIR}/build/lib

export KMP_HW_SUBSET=1t
export KMP_AFFINITY=granularity=fine,compact,1,0

case $spec_override in
  debug)
    echo "debug override spec"
    min_query_count=1000
    min_duration_ms=1
    min_singlestream_query_count=10;
    ;;
  spec)
    echo "use spec"
    min_query_count=24576
    min_duration_ms=60000
    min_singlestream_query_count=1024
    ;;    
  *)
    echo "unknown spec override"
    exit 1
    ;;
esac

case $scenario in
  offline)
    echo "scenario offline"
    loadrun_settings="--min_query_count ${min_query_count} \
                      --min_duration_ms ${min_duration_ms} \
                      --scenario Offline \
                      --sort_samples true \
                      --offline_expected_qps ${target_qps} $@"
    echo ${loadrun_settings}
    ;;
  server)
    echo "scenario server"
    loadrun_settings="--min_query_count ${min_query_count}  \
                     --min_duration_ms ${min_duration_ms} \
                     --scenario Server \
                     --server_target_qps ${target_qps} $@"
    echo ${loadrun_settings}
    ;;
  singlestream)
    echo -n "scenario singlestream"
    loadrun_settings="--min_query_count ${min_singlestream_query_count} \
                     --min_duration_ms ${min_duration_ms} \
                     --scenario SingleStream \
                     $@" 
    echo ${loadrun_settings}
    ;;    
  *)
    echo -n "unknown scenarion"
    exit 1
    ;;
esac
  
numactl -C 56-111 ./cmake-build/loadrun  --cold_run_number 5  --batch_size ${batchsize_i}  \
      -i "/home/user/CK-TOOLS/dataset-imagenet-ilsvrc2012-val/" \
      --class_file "/home/user/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt" \
      --blob "/home/user/models/prepared_blobs/resnet50.zip" \
      --performance_samples $[$2*$1] \
      --total_samples $[$2*$1] \
      --mode PerformanceOnly \
      ${loadrun_settings}
