function run_performance_audit
{
echo "Run TEST01 with performance only with audit"
cp nvidia/TEST01/audit.config .
sed -i "s/accuracy=.*/accuracy=1/" run.sh
sed -i "s/performance=.*/performance=1/" run.sh
sed -i 's/itime=.*/itime=1/g' run.sh
rm -rf results
. ./run.sh TEST01 
}

run_performance_audit

set -x
model=resnet
script=nvidia/TEST01/verify_accuracy.py
folder1=audit
folder2=audit

for model in resnet
do
    for mode in Server Offline
    do
      python $script -p $folder1/$model/$mode/TEST01/performance/run_1/mlperf_log_accuracy.json -a $folder2/$model/$mode/TEST01/accuracy/mlperf_log_accuracy.json 2>&1|tee $folder1/$model/$mode/TEST01/verify_accuracy.txt
    done
done

script=nvidia/TEST01/verify_performance.py
folder1=audit
folder2=submit

for model in resnet
do
for mode in Server Offline
do

   python $script -t $folder1/$model/$mode/TEST01/performance/run_1/mlperf_log_summary.txt -r $folder2/$model/$mode//performance/run_1/mlperf_log_summary.txt 2>&1|tee $folder1/$model/$mode/TEST01/verify_performance.txt

done
done

