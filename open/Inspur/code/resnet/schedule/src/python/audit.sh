#!/bin/bash
audit_dir=./audittest
network="resnet50"
network_tuple=($network)
scenario="Offline"
scenario_tuple=($scenario)
audit="TEST01 TEST03 TEST04-A TEST04-B TEST05"
audit_tuple=($audit)
mode="accuracy performance"
mode_tuple=($mode)
verify_dir=/root/wsh/schedule-benchmark/audittest/audit_release_nvidia_0.5/nvidia/
sub_dir=/root/wsh/schedule-benchmark/results/NF5888M5-gpu/resnet/Offline/
#mkdir /root/wsh/schedule-benchmark/for_audit
#python /root/wsh/schedule-benchmark/audittest/audit_release_nvidia_0.5/nvidia/TEST03/modify_image_data.py -d /root/wsh/mlperf-data/dataset-imagenet-ilsvrc2012-val -o /root/wsh/schedule-benchmark/for_audit --dataset  imagenet
function audit_test_TEST01() {
	for p in $*;
    do
        echo "[$p]";
    done;
    verify_accuracy_dir=$4/$1/$2/TEST01/accuracy
    verify_performance_dir=$4/$1/$2/TEST01/performance/run_1
    cp $5/$3/TEST01/audit.config ./
	python main.py --test-case $1-tf-$2 --config audit.config 
	mv mlperf_log_accuracy.json $verify_accuracy_dir/
	mv mlperf_log_summary.txt $verify_performance_dir/
	mv mlperf_log_detail.txt $verify_performance_dir/
	python $5/TEST01/verify_accuracy.py -a $sub_dir/accuracy/mlperf_log_accuracy.json -p $4/$1/$2/TEST01/accuracy/mlperf_log_accuracy.json | tee $4/$1/$2/TEST01/verify_accuracy.txt
	python $5/TEST01/verify_performance.py -r $sub_dir/performance/run_1/mlperf_log_summary.txt -t $verify_performance_dir/mlperf_log_summary.txt | tee $4/$1/$2/TEST01/verify_performance.txt
}
function audit_test_TEST04-A() {
	for p in $*;
    do
        echo "[$p]";
    done;
    verify_accuracy_dir_A=$4/$1/$2/TEST04-A/accuracy
    verify_performance_dir_A=$4/$1/$2/TEST04-A/performance/run_1
    verify_accuracy_dir_B=$4/$1/$2/TEST04-B/accuracy
    verify_performance_dir_B=$4/$1/$2/TEST04-B/performance/run_1
    cp $5/$3/TEST04-A/audit.config ./
	python main.py --test-case $1-tf-$2 --config audit.config 
	mv mlperf_log_summary.txt $verify_accuracy_dir_A/
	mv mlperf_log_summary.txt $verify_performance_dir_A/
	mv mlperf_log_detail.txt $verify_performance_dir_A/
	cp $5/$3/TEST04-B/audit.config ./
	python main.py --test-case $1-tf-$2 --config audit.config 
	mv mlperf_log_summary.txt $verify_accuracy_dir_B/
	mv mlperf_log_summary.txt $verify_performance_dir_B/
	mv mlperf_log_detail.txt $verify_performance_dir_B/
	python $5/$3/verify_test4_performance.py -u $verify_performance_dir_A/mlperf_log_summary.txt -s verify_performance_dir_B/mlperf_log_summary.txt | tee $4/$1/$2/TEST04-A/verify_performance.txt

}
function audit_test_TEST05() {
	for p in $*;
    do
        echo "[$p]";
    done;
    verify_accuracy_dir=$4/$1/$2/TEST05/accuracy
    verify_performance_dir=$4/$1/$2/TEST05/performance/run_1
    cp $5/$3/TEST05/audit.config ./
	python main.py --test-case $1-tf-$2 --config audit.config | tee accuracy.txt

	mv mlperf_log_accuracy.json $verify_accuracy_dir/
	mv mlperf_log_summary.txt $verify_performance_dir/
	mv mlperf_log_detail.txt $verify_performance_dir/
	python $5/$3/verify_performance.py -r $sub_dir/performance/run_1/mlperf_log_summary.txt -t $verify_performance_dir/mlperf_log_summary.txt | tee $4/$1/$2/TEST05/verify_performance.txt

}
function audit_test_TEST03() {
	for p in $*;
    do
        echo "[$p]";
    done;
    verify_accuracy_dir=$4/$1/$2/TEST03/accuracy
    verify_performance_dir=$4/$1/$2/TEST03/performance/run_1
	python main.py --test-case $1-tf-$2 --dataset_path /root/wsh/schedule-benchmark/for_audit | tee accuracy.txt
	mv accuracy.txt $verify_accuracy_dir/
	mv mlperf_log_summary.txt $verify_performance_dir/
	mv mlperf_log_detail.txt $verify_performance_dir/
	mv mlperf_log_accuracy.json $verify_accuracy_dir/
	python $5/$3/verify_performance.py -r $sub_dir/performance/run_1/mlperf_log_summary.txt -t $4/$1/$2/TEST03/performance/run_1/mlperf_log_summary.txt | tee $4/$1/$2/TEST03/verify_performance.txt

}
for((network_index=0;network_index<${#network_tuple[@]};network_index++)); do
	if [ -e ${audit_dir}/${network_tuple[network_index]} ];then
		echo "have no ${audit_dir}/${network_tuple[network_index]}"
	else 
		mkdir -p ${audit_dir}/${network_tuple[network_index]}
	fi
	for((scenario_index=0;scenario_index<${#scenario_tuple[@]};scenario_index++)); do
		if [ -e ${audit_dir}/${network_tuple[network_index]}/${scenario_tuple[scenario_index]} ];then
			echo "have no ${audit_dir}/${network_tuple[network_index]}/${scenario_tuple[scenario_index]}"
		else 
			mkdir -p ${audit_dir}/${network_tuple[network_index]}/${scenario_tuple[scenario_index]}
		fi
		for((audit_index=0;audit_index<${#audit_tuple[@]};audit_index++)); do
			if [ -e ${audit_dir}/${network_tuple[network_index]}/${scenario_tuple[scenario_index]}/${audit_tuple[audit_index]} ];then
				echo "have no ${audit_dir}/${network_tuple[network_index]}/${scenario_tuple[scenario_index]}/${audit_tuple[audit_index]}"
			else 
				mkdir -p ${audit_dir}/${network_tuple[network_index]}/${scenario_tuple[scenario_index]}/${audit_tuple[audit_index]}
			fi
			mkdir -p ${audit_dir}/${network_tuple[network_index]}/${scenario_tuple[scenario_index]}/${audit_tuple[audit_index]}/accuracy
			mkdir -p ${audit_dir}/${network_tuple[network_index]}/${scenario_tuple[scenario_index]}/${audit_tuple[audit_index]}/performance/run_1
			#for((mode_index=0;mode_index<2;mode_index++)); do
				#if [ -e ${audit_dir}/${network_tuple[network_index]}/${scenario_tuple[scenario_index]}/${audit_tuple[audit_index]}/${mode_tuple[mode_index]} ];then
					#echo "have no ${audit_dir}/${network_tuple[network_index]}/${scenario_tuple[scenario_index]}/${audit_tuple[audit_index]}/${mode_tuple[mode_index]}"
			#	else 
				#	mkdir -p ${audit_dir}/${network_tuple[network_index]}/${scenario_tuple[scenario_index]}/${audit_tuple[audit_index]}/${mode_tuple[mode_index]}
				#fi
		    
			#done
		done
		audit_test_TEST01 ${network_tuple[network_index]} ${scenario_tuple[scenario_index]} ${audit_tuple[audit_index]}  ${audit_dir} ${verify_dir}
		audit_test_TEST03 ${network_tuple[network_index]} ${scenario_tuple[scenario_index]} ${audit_tuple[audit_index]}  ${audit_dir} ${verify_dir}
		audit_test_TEST04-A ${network_tuple[network_index]} ${scenario_tuple[scenario_index]} ${audit_tuple[audit_index]}  ${audit_dir} ${verify_dir}
		audit_test_TEST05 ${network_tuple[network_index]} ${scenario_tuple[scenario_index]} ${audit_tuple[audit_index]}  ${audit_dir} ${verify_dir}
		#	audit_test ${network_tuple[network_index]} ${scenario_tuple[scenario_index]} ${audit_tuple[audit_index]} ${mode_tuple[mode_index]} ${audit_dir} ${verify_dir}
	done
done





