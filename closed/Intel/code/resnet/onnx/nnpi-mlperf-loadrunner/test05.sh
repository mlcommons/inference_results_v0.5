#sed -i 's/--images \".*\" \\/--images \"\/home\/mlt\/tools\/val/\" \\/g' netrun.sh
#sed -i 's/--images \".*\" \\/--images \"\/home\/mlt\/tools\/val\/\" \\/g' loadrun.sh
#sed -i "s/accuracy=.*/accuracy=0/" run.sh
#sed -i "s/performance=.*/performance=1/" run.sh
#sed -i 's/itime=.*/itime=1/g' run.sh

run_test05(){
cp nvidia/TEST05/audit.config .
rm -rf results
. run.sh TEST05
}

run_test05

set -x
script=nvidia/TEST05/verify_performance.py
folder1=submit
folder2=audit

for model in resnet
do
    for mode in Server Offline
    do
       python $script -r ${folder1}/${model}/${mode}/performance/run_1/mlperf_log_summary.txt -t ${folder2}/${model}/${mode}/TEST05/performance/run_1/mlperf_log_summary.txt 2>&1|tee ${folder2}/${model}/${mode}/TEST05/verify_performance.txt
    done
done

set +x

