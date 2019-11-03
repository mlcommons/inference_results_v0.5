
audit=$1
accuracy=0
performance=1
itime=1
export SCHEDULER=RR


run_mlperf(){

mode=$1
model=$2

if [ "$accuracy" == "1" ];then

   if [ "${mode}" == "server" ];then
      . clean.sh;sh loadrun.sh 2 ${mode} 28 resnet50 112 AccuracyOnly ${audit}
   else
      . clean.sh;sh loadrun.sh 16 ${mode} 28 resnet50 112 AccuracyOnly ${audit}
   fi

   wait
fi

if [ "$performance" == "1" ];then

for((i=1;$i<=$itime;i=$(($i + 1))));  
do
   rm mlperf_log*
   if [ "${mode}" == "server" ];then
      if [ "${model}" == "resnet50" ];then
         . clean.sh;sh loadrun.sh 2 ${mode} 28 resnet50 112 PerformanceOnly ${audit}
      fi
   else
      if [ "${model}" == "resnet50" ];then
         . clean.sh;sh loadrun.sh 16 ${mode} 28 resnet50 112 PerformanceOnly ${audit}
      fi
   fi
   wait

   if [ "${mode}" == "server" ];then
      mode="Server"
   fi
   if [ "${mode}" == "offline" ];then
      mode="Offline"
   fi

   if [ "${model}" == "resnet50" ];then
      model=resnet
   fi
 
   mkdir -p results/${model}/${mode}/performance/run_${i}
   mv results/${model}/${mode}/performance/mlperf_log_* results/${model}/${mode}/performance/run_${i} 

done

if [[ ${audit} =~ ^TEST ]];then
   mkdir -p audit/${model}/${mode}/${audit}/performance/run_1
   mv audit/${model}/${mode}/${audit}/performance/mlperf_log_* audit/${model}/${mode}/${audit}/performance/run_1/.
fi

fi
}

run_mlperf server resnet50
run_mlperf offline resnet50

