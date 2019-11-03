#~/bin/bash

# parse command line

function print_usage () {
	echo 'Usage: ./offline_scenario.sh [-i|--input_data <path to input data>] [-c|--class_file <path to classification file>] [-b|--blob_file <path to the compiled blob>] [-o|--output_dir] [--num_devices] [--num_dev_nets_per_device] [--num_parallel_infers]'
	echo '-i|--input_data		      (REQUIRED) path to input data'
	echo '-c|--class_file		      (REQUIRED) files to check classification res'
	echo '-b|--blob_file		      (REQUIRED) path to the compiled blob'	
	echo '-o|--output_dir		      (OPTIONAL) path to output directory, default is ./output_dir'
	echo '--num_devices		        (OPTIONAL) number of devices to use, default is 1'
  echo '--num_dev_nets_per_device		(OPTIONAL) number of ICE cores to use per device, default is 12'
  echo '--num_parallel_infers		(OPTIONAL) number of infers to use on every ICE core, default is 2'
  echo '-h|--help				Display this help message'	
	exit 1
}

OPTS=`getopt -o hi:c:b:o: --long help,input_data:,class_file:,blob_file:,output_dir:,num_devices:,num_dev_nets_per_device:,num_parallel_infers: -n 'parse-options' -- "$@"`
if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; print_usage ; fi
eval set -- "$OPTS"

INPUT_DATA=''
CLASS_FILE=''
BLOB_PATH=''
OUTPUT_DIR='output_dir'
SCENARIO='offline'
NUM_DEVICES=1
NUM_DEV_NETS=12
NUM_INFERS=2

while true; do
  case "$1" in
    -i | --input_data ) INPUT_DATA="$2"; shift; shift ;;
    -c | --class_file ) CLASS_FILE="$2"; shift; shift ;;
    -b | --blob_file ) BLOB_PATH="$2"; shift; shift ;;
	  -o |--output_dir ) OUTPUT_DIR="$2"; shift; shift ;;
	  --num_devices ) NUM_DEVICES=$2; shift; shift ;;
    --num_dev_nets_per_device ) NUM_DEV_NETS=$2; shift; shift ;;
	  --num_parallel_infers ) NUM_INFERS=$2; shift; shift ;;
    -h | --help ) print_usage ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if [ "$INPUT_DATA" == '' ]; then
	echo 'Error - did not receive input data'
	print_usage
fi
if [ "$CLASS_FILE" == '' ]; then
	echo 'Error - did not receive class file'
	print_usage
fi
if [ "$BLOB_PATH" == '' ]; then
	echo 'Error - did not receive blob file'
	print_usage
fi
if [ ! -d "$INPUT_DATA" ];then
	if [ ! -f "$INPUT_DATA" ]; then
		echo 'Error - path to input data does not exists!'
		exit 1
	fi
fi
if [ ! -f "$CLASS_FILE" ];then
	echo 'Error - path to class file does not exists!'
	exit 1
fi
if [ ! -f "$BLOB_PATH" ];then
	echo 'Error - path to blob file does not exists!'
	exit 1
fi
if [ ! -d "$OUTPUT_DIR" ];then
	mkdir "$OUTPUT_DIR"
fi

if [ ! -d "$OUTPUT_DIR/$SCENARIO" ];then
	mkdir "$OUTPUT_DIR/$SCENARIO"
fi

set -x

# CAFFE2_DIR=../../

# export LD_PRELOAD=/opt/intel/compilers_and_libraries/linux/lib/intel64/libiomp5.so

export CAFFE2_INFERENCE_MEM_OPT=1

export OMP_NUM_THREADS=56  KMP_AFFINITY="proclist=[0-55],granularity=fine,explicit"

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CAFFE2_DIR}/build/lib

export KMP_HW_SUBSET=1t
export KMP_AFFINITY=granularity=fine,compact,1,0

./cmake-build/loadrun \
      -i "$INPUT_DATA" \
      --class_file "$CLASS_FILE" \
      --label_file "$CLASS_FILE" \
      --blob "$BLOB_PATH" \
	  --batch_size 1 \
      --min_query_count 1 \
      --min_duration_ms 60000 \
      --performance_samples 1000 \
      --total_samples 50000 \
      --scenario Offline \
     --offline_expected_qps 1.0 \
      --mode PerformanceOnly \
      --card_num "$NUM_DEVICES" \
      --dev_nets_per_card "$NUM_DEV_NETS" \
      --parallel_infer_count "$NUM_INFERS" \
	  -w 5 \
      -n 1 \
      -c 5 \
      -s       
    
set +x

mv "mlperf_log_trace.json" "$OUTPUT_DIR/$SCENARIO"
mv "mlperf_log_accuracy.json" "$OUTPUT_DIR/$SCENARIO"
mv "mlperf_log_summary.txt" "$OUTPUT_DIR/$SCENARIO"
mv "mlperf_log_detail.txt" "$OUTPUT_DIR/$SCENARIO"

