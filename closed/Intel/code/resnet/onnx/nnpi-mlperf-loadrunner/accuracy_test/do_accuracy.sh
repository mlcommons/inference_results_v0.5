source d_r_s.sh
python3 accuracy-imagenet.py --mlperf-accuracy-file ../output_dir/server/mlperf_log_accuracy.json --imagenet-val-file val.txt --dtype int32 --verbose
