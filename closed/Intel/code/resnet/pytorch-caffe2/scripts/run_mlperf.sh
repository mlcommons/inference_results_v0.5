#!/bin/bash

# environment
EULER_ROOT=../../third_party/ideep/euler

export LD_PRELOAD=$EULER_ROOT/lib/libiomp5.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64
export LD_LIBRARY_PATH=../../build/lib:$LD_LIBRARY_PATH
export KMP_HW_SUBSET=1t
export KMP_AFFINITY=granularity=fine,compact,1,0
export CAFFE2_INFERENCE_MEM_OPT=1

bin_file=./inferencer
cmd_file=__command.sh
images="/lustre/dataset/imagenet/img_raw/ILSVRC2012_img_val/"
labels="val.txt"
random_multibatch=false
shared_input="USE_LOCAL"
shared_weight="USE_LOCAL"
proto_txt=""

# query platform info
ncores_per_socket=
nthreads_per_core=1
nthreads=$NTHREADS
nsockets=$( lscpu | grep 'Socket(s)' | cut -d: -f2 )
ncores_per_socket=${ncores_per_socket:=$( lscpu | grep 'Core(s) per socket' | cut -d: -f2 )}
nthreads_max=$(( nsockets  * ncores_per_socket * nthreads_per_core ))
nthreads=${nthreads:=$nthreads_max}

# default options
model=resnet50
batch_size=32
instances=4
iterations=64
log_level=0
share_memory=0
engine="euler"

show_help() {
cat <<!
$0 [options]
-t,--model            Model. resnet50|mobilenetv1
-b,--batch-size       Batch-size
-i,--instances        Number of instances
-r,--iterations       Number of iterations
-E,--engine           Engine. mkldnn|euler
-s,--share-memory     Share memory: 0|1
-m,--images           Images folder
-a,--labels           Label file
-l,--log-level        Log level. -2:verbose; -1: verbose++; 0:silence
-p,--proto-txt        Proto txt model
-h,--help             This page

Environment:
NTHREADS=...          Number of thread to be used.

!
}

OPTIND=1
while getopts ":t:b:r:i:E:l:s:m:p:a:h-:" opt; do
  case "$opt" in
    t) model=$OPTARG ;;
    b) batch_size=$OPTARG ;;
    i) instances=$OPTARG ;;
    r) iterations=$OPTARG ;;
    E) engine=$OPTARG ;;
    l) log_level=$OPTARG ;;
    m) images=$OPTARG ;;
    a) labels=$OPTARG ;;
    s) share_memory=$OPTARG ;;
    p) proto_txt=$OPTARG ;;
    h) show_help && exit 0 ;;
    -)
    case "${OPTARG}" in
      model=*) engine=${OPTARG#*=} ;;
      batch-size=*) batch_size=${OPTARG#*=} ;;
      instances=*) instances=${OPTARG#*=} ;;
      iterations=*) iterations=${OPTARG#*=} ;;
      engine=*) engine=${OPTARG#*=} ;;
      images*) images=${OPTARG#*=} ;;
      labels*) labels=${OPTARG#*=} ;;
      log-level=*) log_level=${OPTARG#*=} ;;
      share-memory=*) share_memory=${OPTARG#*=} ;;
      proto_txt=*) proto_txt=${OPTARG#*=} ;;
      help) show_help && exit 0 ;;
    esac
    ;;
  esac
done
shift $((OPTIND-1))

mt_runtime="OMP"
if [ x"$engine" = "xeuler" ]; then
  if EULER_VERBOSE=1 $EULER_ROOT/build/tests/elt_conv -version | grep -i 'MT_RUNTIME: TBB' >& /dev/null; then
    mt_runtime="TBB"
  fi
  if [ x"$model" = "xresnet50" ]; then  
    init_txt=../models/resnet50/init_net_int8_euler.pb
    if [ "x$proto_txt" = "x" ]; then
      if [ $batch_size -lt 8 ] && [ $nthreads -gt 12 ]; then
        if [ x"$mt_runtime" = "xTBB" ]; then
          proto_txt=../models/resnet50/predict_net_int8_euler_lat_lazy.pbtxt
        else
          proto_txt=../models/resnet50/predict_net_int8_euler_lat.pbtxt
        fi
      else
        if [ x"$mt_runtime" = "xTBB" ]; then
          proto_txt=../models/resnet50/predict_net_int8_euler_lazy.pbtxt
        else
          proto_txt=../models/resnet50/predict_net_int8_euler.pbtxt
        fi
      fi
    fi
  else
    init_txt=../models/mobilenet/init_net_int8_euler.pb
    if [ "x$proto_txt" = "x" ]; then
      if [ $batch_size -lt 8 ] && [ $nthreads -gt 12 ]; then
        if [ x"$mt_runtime" = "xTBB" ]; then
          proto_txt=../models/mobilenet/predict_net_int8_euler_lat_lazy.pbtxt
        else
          proto_txt=../models/mobilenet/predict_net_int8_euler_lat.pbtxt
        fi
      else
        if [ x"$mt_runtime" = "xTBB" ]; then
          proto_txt=../models/mobilenet/predict_net_int8_euler_lazy.pbtxt
        else
          proto_txt=../models/mobilenet/predict_net_int8_euler.pbtxt
        fi
      fi
    fi
  fi  
  export U8_INPUT_OPT=1
elif [ x"$engine" = "xmkldnn" ]; then
  if [ x"$model" = "xresnet50" ]; then  
    init_txt=../models/resnet50/init_net_int8.pb
    if [ "x$proto_txt" = "x" ]; then
      proto_txt=../models/resnet50/predict_net_int8.pbtxt
    fi
  else
    init_txt=../models/mobilenet/init_net_int8.pb
    if [ "x$proto_txt" = "x" ]; then
      proto_txt=../models/mobilenet/predict_net_int8.pbtxt
    fi
  fi
else
  echo "Error: Invalid engine. Expect 'mkldnn' or 'euler'"
  exit -1
fi

if [ x$log_level = "x-1" ]; then
  export MKLDNN_VERBOSE=1
  export EULER_VERBOSE=1
fi

nthreads_per_instance=$((nthreads / instances))
if [ $nthreads -gt $nthreads_max ]; then
  echo "Warning: core oversubscribed:" \
       "nthreads ($nthreads) > Max HW thread number ($nthreads_max)"

elif (( $nthreads % $instances)); then
  echo "Warning: core undersubsribed:" \
       "threads-per-instance " \
       "($nthreads_per_instance) * instances ($instances) < ncores ($nthreads)" 
fi

# garbage collection
rm -f $cmd_file .euler_key_* logs/_run_mlperf_*.log
mkdir -p logs

gen_cmd() {
  local _numa_id=$1
  local _nthreads_per_instance=$2
  local _proc_start=$3
  local _proc_end=$4
  local _shared_input=$5
  local _shared_weight=$6
  local _logfile=$7

  if [ x"$mt_runtime" = "xOMP" ]; then
    THREAD_BINDING="KMP_AFFINITY=\"proclist=[$_proc_start-$_proc_end],granularity=fine,explicit\" "
  else
    THREAD_BINDING="KMP_BLOCKTIME=0 numactl -l -C $_proc_start-$_proc_end "
  fi

echo "EULER_SHARED_WORKSPACE=$share_memory EULER_NUMA_NODE=$_numa_id \
OMP_NUM_THREADS=$_nthreads_per_instance \
$THREAD_BINDING \
$bin_file --w 20 --quantized true \
--batch_size $batch_size \
--iterations $iterations \
--device_type ideep \
--images $images \
--labels $labels \
--random_multibatch $random_multibatch \
--numa_id $numa_id \
--log_level $log_level \
--init_net_path $init_txt \
--predict_net_path $proto_txt \
--shared_memory_option $_shared_input \
--shared_weight $_shared_weight \
--net_conf $model \
--data_order NHWC \
--dummy_data false \
> $_logfile 2>&1 &" >>temp.sh
  
  cat <<!
EULER_SHARED_WORKSPACE=$share_memory EULER_NUMA_NODE=$_numa_id \
OMP_NUM_THREADS=$_nthreads_per_instance \
$THREAD_BINDING \
$bin_file --w 20 --quantized true \
--batch_size $batch_size \
--iterations $iterations \
--device_type ideep \
--images $images \
--labels $labels \
--random_multibatch $random_multibatch \
--numa_id $numa_id \
--log_level $log_level \
--init_net_path $init_txt \
--predict_net_path $proto_txt \
--shared_memory_option $_shared_input \
--shared_weight $_shared_weight \
--net_conf $model \
--data_order NHWC \
--dummy_data false \
> $_logfile 2>&1 &
!
}

# share memory
if [ x"$share_memory" = "x2" ]; then
  shared_input="USE_SHM"
  shared_weight="USE_SHM"
  for numa_id in `seq 0 $((nsockets - 1))`; do
    proc_start=$((ncores_per_socket * numa_id))
    proc_end=$((ncores_per_socket * (numa_id + 1) - 1))
    eval "$(gen_cmd $numa_id $ncores_per_socket $proc_start $proc_end \
      "CREATE_USE_SHM" "CREATE_USE_SHM" /dev/null)"
  done
  wait
fi

# generate command
for inst_id in `seq $instances`; do
  proc_start=$((nthreads_per_instance * (inst_id - 1)))
  proc_end=$((nthreads_per_instance * inst_id - 1))
  numa_id=$((proc_start / ncores_per_socket))
  gen_cmd $numa_id $nthreads_per_instance $proc_start $proc_end \
    $shared_input $shared_weight \
    logs/_run_mlperf_${inst_id}.log >> $cmd_file
done
echo wait >> $cmd_file

# execute
eval "$(cat $cmd_file)"

# report results
images=$(( iterations * batch_size ))
fps=$(grep 'hardware time' logs/_run_mlperf_*.log | \
  awk -v images=$images '{ fps += images / $NF } END { print fps }')
latency=$(grep 'hardware time' logs/_run_mlperf_*.log | \
  awk -v images=$images -v instances=$instances \
  '{ latency_sum += 1000 * $NF / images } END { print latency_sum / instances }')

echo -----*------*-----
echo Threads: $nthreads
echo Engine: "$engine($mt_runtime)"
echo Instances: $instances
echo Threads per instance: $nthreads_per_instance
echo Iterations: $iterations
echo Batch size: $batch_size
echo Average latency for one image: ${latency}ms
echo FPS: $fps images/sec
echo Accuracy: $(cat logs/_run_mlperf_*.log | grep top1 | cut -d' ' -f3- | uniq)
