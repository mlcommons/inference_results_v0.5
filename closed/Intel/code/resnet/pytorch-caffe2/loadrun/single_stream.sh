#~/bin/bash
set -x

model=$1
mode=$2
audit=$3

net_conf=${model}

if [ ${model} == "mobilenet" ];then
   net_conf=mobilenetv1
fi

set -x

CAFFE2_DIR=../../

#export SCHEDULER=RR

export LD_PRELOAD=../../third_party/ideep/euler/lib/libiomp5.so

export U8_INPUT_OPT=1

export CAFFE2_INFERENCE_MEM_OPT=1

export OMP_NUM_THREADS=28

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CAFFE2_DIR}/build/lib:../../third_party/ideep/euler/lib:/opt/intel/compilers_and_libraries_2019/linux/lib/intel64

export KMP_HW_SUBSET=1t
export KMP_AFFINITY=granularity=fine,compact,1,0
export CAFFE2_USE_EULER=ON

if [ ${model} == "mobilenet" ];then
  net_conf="mobilenetv1"
else
  net_conf="resnet50"
fi

if [ "${model}" == "mobilenet" ];then
echo "run mobilenet ......."
echo "************************************"
EULER_SHARED_WORKSPACE=0 EULER_NUMA_NODE=0 numactl -C 0-27 -l ./loadrun --w 5 --quantized true \
      --batch_size 1 \
      --iterations 50000 \
      --device_type "ideep" \
      --images "/lustre/dataset/imagenet/img_raw/ILSVRC2012_img_val/" \
      --labels "val.txt" \
      --init_net_path ../models/${model}/init_net_int8_euler.pb \
      --predict_net_path ../models/${model}/predict_net_int8_euler_lat.pbtxt \
      --shared_memory_option USE_LOCAL \
      --shared_weight USE_LOCAL \
      --min_query_count 1024 \
      --min_duration_ms 60000 \
      --performance_samples 50000 \
      --single_stream_expected_latency_ns 500000 \
      --total_samples 50000 \
      --scenario SingleStream \
      --net_conf ${net_conf} \
      --schedule_local_batch_first false \
      --include_accuracy true \
      --data_order NHWC \
      --model_name ${model} \
      --mode ${mode}

     # --mlperf_config_file_name mlperf.conf \
     # --user_config_file_name user.conf.${model} \
else
EULER_SHARED_WORKSPACE=0 EULER_NUMA_NODE=0 numactl -C 0-27 -l ./loadrun --w 5 --quantized true \
      --batch_size 1 \
      --iterations 50000 \
      --device_type "ideep" \
      --images "/lustre/dataset/imagenet/img_raw/ILSVRC2012_img_val/" \
      --labels "val.txt" \
      --init_net_path ../models/${model}/init_net_int8_euler.pb \
      --predict_net_path ../models/${model}/predict_net_int8_euler_lat.pbtxt \
      --shared_memory_option USE_LOCAL \
      --shared_weight USE_LOCAL \
      --min_query_count 1024 \
      --min_duration_ms 60000 \
      --performance_samples 50000 \
      --total_samples 50000 \
      --scenario SingleStream \
      --net_conf ${net_conf} \
      --schedule_local_batch_first false \
      --mlperf_config_file_name mlperf.conf \
      --user_config_file_name user.conf.${model} \
      --include_accuracy true \
      --data_order NHWC \
      --model_name ${model} \
      --mode ${mode}
fi 

if [ ${model} == "resnet50" ];then
   model="resnet"
fi

mkdir -p results/${model}/SingleStream/performance/run_1
mkdir -p results/${model}/SingleStream/accuracy
if [ "${audit}" != "" ];then
mkdir -p audit/${model}/SingleStream/${audit}/performance/run_1
mkdir -p audit/${model}/SingleStream/${audit}/accuracy
fi


if [ "${mode}" == "PerformanceOnly" ];then

cp mlperf_log_detail.txt mlperf_log_summary.txt results/${model}/SingleStream/performance/run_1/.
if [ -f mlperf_log_accuracy.json ];then
cp mlperf_log_accuracy.json results/${model}/SingleStream/performance/run_1/.
fi

if [[ ! -z "${audit}" ]];then
cp mlperf_log_detail.txt mlperf_log_summary.txt audit/${model}/SingleStream/${audit}/performance/run_1/.
if [ -f mlperf_log_accuracy.json ];then
cp mlperf_log_accuracy.json audit/${model}/SingleStream/${audit}/performance/run_1/.
fi
fi

fi

if [ "${mode}" == "AccuracyOnly" ];then
cp mlperf_log_detail.txt mlperf_log_summary.txt mlperf_log_accuracy.json results/${model}/SingleStream/accuracy
if [[ ! -z "${audit}" ]];then
cp mlperf_log_detail.txt mlperf_log_summary.txt mlperf_log_accuracy.json audit/${model}/SingleStream/${audit}/accuracy
fi
fi










