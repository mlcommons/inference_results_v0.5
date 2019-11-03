#~/bin/bash
set -x

CAFFE2_DIR=../../../pytorch

#$1 batch size
#$2 iterations
#$3 the processor list
#$4 the OMP thread count
#$5 instance name
#$6 server name
#$7 shared mem option
#$8 numa id
#$9 shared weight option
#$10 model 
#$11 measure_mode
#$12 scenario

scenario=${12} 

schedule_local_batch_first=true
sync_by_cond=false

if [ "${SCHEDULER}" = "LBF" ]; then
  schedule_local_batch_first=true
  echo "schedule_local_batch_first true"
fi

if [ "${SCHEDULER}" = "RR" ]; then
  schedule_local_batch_first=false
  echo "schedule_local_batch_first false"
fi

if [ "${SYNC_BY_COND}" = "true" ]; then
  sync_by_cond=true
  echo "sync_by_cond true" 
fi

pbtxt=../models/${10}/predict_net_int8_euler.pbtxt
case $scenario in
  offline)
    echo "scenario offline"
    netrun_settings="--schedule_local_batch_first ${schedule_local_batch_first} \
                     --sync_by_cond ${sync_by_cond}"
    echo ${netrun_settings}
    pbtxt=../models/${10}/predict_net_int8_euler.pbtxt
    ;;
  server)
    echo "scenario server"
    netrun_settings="--schedule_local_batch_first false \
                     --sync_by_cond false"
    if [ "${10}" == "resnet50" ];then
       pbtxt=../models/${10}/predict_net_int8_euler_lat.pbtxt
    fi
    echo ${netrun_settings}
    ;;
  singlestream)
    echo -n "scenario singlestream"
    netrun_settings="--schedule_local_batch_first false \
                     --sync_by_cond false" 
    pbtxt=../models/${10}/predict_net_int8_euler_lat.pbtxt
    echo ${netrun_settings}
    ;;    
  *)
    echo -n "unknown scenarion"
    exit 1
    ;;
esac


export CAFFE2_INFERENCE_MEM_OPT=1

# feel free to use numacrl or taskset to control affinity
export OMP_NUM_THREADS=$4 

export KMP_HW_SUBSET=1t

export KMP_AFFINITY=granularity=fine,compact,1,0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CAFFE2_DIR}/build/lib:../../third_party/ideep/euler/lib:/opt/intel/compilers_and_libraries_2019/linux/lib/intel64

export LD_PRELOAD=../../third_party/ideep/euler/lib/libiomp5.so

export U8_INPUT_OPT=1

export CAFFE2_USE_EULER=ON

if [ ${10} == "mobilenet" ];then
  net_conf="mobilenetv1"
else
  net_conf="resnet50"
fi


# batchsize_i * iteration_i decides how many imgs will be loaded to ram
echo EULER_SHARED_WORKSPACE=0 EULER_NUMA_NODE=$8 numactl -C $3 ./netrun  --w 5 --quantized true --batch_size $1 --iterations $2 \
      --device_type "ideep" \
      --images "/lustre/dataset/imagenet/img_raw/ILSVRC2012_img_val/" \
      --labels "val.txt" \
      --init_net_path ../models/${10}/init_net_int8_euler.pb \
      --predict_net_path ${pbtxt} \
      --random_multibatch true \
      --shared_memory_option $7 \
      --shared_weight $9 \
      --numa_id $8 \
      --instance $5 \
      --server $6 \
      --mode ${11} \
      --include_accuracy true \
      --data_order NHWC \
      --net_conf $net_conf \
      ${netrun_settings}

EULER_SHARED_WORKSPACE=0 EULER_NUMA_NODE=$8 numactl -C $3 ./netrun  --w 5 --quantized true --batch_size $1 --iterations $2 \
      --device_type "ideep" \
      --images "/lustre/dataset/imagenet/img_raw/ILSVRC2012_img_val/" \
      --labels "val.txt" \
      --init_net_path ../models/${10}/init_net_int8_euler.pb \
      --predict_net_path ${pbtxt} \
      --random_multibatch true \
      --shared_memory_option $7 \
      --shared_weight $9 \
      --numa_id $8 \
      --instance $5 \
      --server $6 \
      --mode ${11} \
      --include_accuracy true \
      --data_order NHWC \
      --net_conf $net_conf \
      ${netrun_settings}










