#~/bin/bash

CAFFE2_DIR=../../
batchsize_i=1
iteration_i=10000
shared_memory_option="CREATE_USE_SHM"
shared_weight="USE_LOCAL"
numa_id=0
random_multibatch=true
log_level=0
echo "Usage: $0  batchsize iteration log_level shared_memory_option numa_id random_multibatch
             example: $0 1 1000 0 USE_SHM 0 true
             or use default configuration: $0"
if (($# > 1)); then
  batchsize_i=$1
  iteration_i=$2
fi
# log_level -1 means profiling, -2 means iteration time, 0 means INFO, 2 means error, 3 means FATAL
if (($# > 2)); then
  log_level=$3
fi
# shared_memory_option USE_SHM means use shared memory USE_LOCAL means use local memory
# FREE_USE_SHM means free shared memory after this process end
# CREATE_USE_SHM means create new shared memory at the beginnig of this process
if (($# > 3)); then
  if (($4 == "USE_SHM" || "USE_LOCAL" || "FREE_USE_SHM" || "CREATE_USE_SHM")); then
    shared_memory_option=$4
  else
    echo "wrong shared memory option!
          only supported: 'USE_LOCAL', 'USE_SHM' , 'FREE_USE_SHM', 'CREATE_USE_SHM'"
    exit
  fi
fi
if (($# > 4)); then
  numa_id=$5
fi
# random_multibatch option true means use random multi batch input, false will use continuos input
if (($# > 5)); then
  random_multibatch=$6
fi

export LD_PRELOAD=/opt/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64/libiomp5.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64
#export LD_PRELOAD=/opt/intel/compilers_and_libraries_2019.1.144/linux/compiler/lib/intel64/libiomp5.so
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.1.144/linux/compiler/lib/intel64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CAFFE2_DIR}/build/lib

#export U8_INPUT_OPT=1
export CAFFE2_INFERENCE_MEM_OPT=1
export OMP_NUM_THREADS=28  KMP_AFFINITY="proclist=[0-27],granularity=fine,explicit"
#export OMP_NUM_THREADS=96  KMP_AFFINITY="proclist=[0-95],granularity=fine,explicit"

if (($log_level == -1)); then
  export MKLDNN_VERBOSE=1
  export EULER_VERBOSE=1
fi

./inferencer  --w 5 --quantized true --batch_size ${batchsize_i} --iterations ${iteration_i} \
      --device_type "ideep" \
      --images "/lustre/dataset/imagenet/img_raw/ILSVRC2012_img_val/" \
      --labels "val.txt" \
      --random_multibatch ${random_multibatch} \
      --shared_memory_option ${shared_memory_option} \
      --numa_id ${numa_id} \
      --log_level ${log_level} \
      --init_net_path ../models/resnet50/init_net_int8.pb \
      --predict_net_path ../models/resnet50/predict_net_int8.pbtxt \
      --net_conf resnet50 \
      --shared_weight ${shared_weight} \
      --data_order NHWC \
      # --dummy_data true \
      # --data_order "NHWC" \
