#sed -i 's/--images \".*\" \\/--images \"\/home\/mlt\/tools\/val/\" \\/g' netrun.sh
#sed -i 's/--images \".*\" \\/--images \"\/home\/mlt\/tools\/val\/\" \\/g' loadrun.sh
#sed -i "s/accuracy=.*/accuracy=0/" run.sh
#sed -i "s/performance=.*/performance=1/" run.sh
#sed -i 's/itime=.*/itime=1/g' run.sh


run_test04(){
cp nvidia/TEST04-A/audit.config .
. run.sh TEST04-A

}

run_test04-B(){
cp nvidia/TEST04-B/audit.config .
. run.sh TEST04-B
}

run_test04
run_test04-B

set -x
script=nvidia/TEST04-A/verify_test4_performance.py
folder1=audit
folder2=audit

for model in resnet
do
for mode in Server Offline
do
      python $script -u ${folder1}/${model}/${mode}/TEST04-A/performance/run_1/mlperf_log_summary.txt -s ${folder2}/${model}/${mode}/TEST04-B/performance/run_1/mlperf_log_summary.txt 2>&1|tee ${folder1}/${model}/${mode}/TEST04-A/verify_test4_performance.txt
done
done

set +x

