run_test03(){
sed -i 's/--images \".*\" \\/--images \"\/home\/mlt\/tools\/val_new\/imagenet\/\" \\/g' netrun.sh
sed -i 's/--images \".*\" \\/--images \"\/home\/mlt\/tools\/val_new\/imagenet\/\" \\/g' loadrun.sh

if [ -f "audit.conf" ];
then
  rm audit.config
fi
sed -i "s/accuracy=.*/accuracy=1/" run.sh
sed -i "s/performance=.*/performance=1/" run.sh
sed -i 's/itime=.*/itime=1/g' run.sh
. run.sh TEST03

}

run_test03

set -x
script=nvidia/TEST03/verify_performance.py
folder1=audit
folder2=submit

for model in resnet
do
   for mode in Server Offline
   do
       python $script -t $folder1/$model/$mode/TEST03/performance/run_1/mlperf_log_summary.txt -r $folder2/$model/$mode/performance/run_1/mlperf_log_summary.txt 2>&1|tee $folder1/$model/$mode/TEST03/verify_performance.txt
   done
done

set +x

