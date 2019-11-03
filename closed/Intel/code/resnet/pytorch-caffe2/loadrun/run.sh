
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
      if [ "${model}" == "resnet50" ];then
         . clean.sh;sh run_loadrun.sh 2 ${mode} 28 resnet50 112 AccuracyOnly ${audit}
      else
         . clean.sh;sh run_loadrun.sh 4 ${mode} 28 mobilenet 112  AccuracyOnly ${audit}
      fi
   else
      if [ "${model}" == "resnet50" ];then
         . clean.sh;sh run_loadrun.sh 16 ${mode} 28 resnet50 112 AccuracyOnly ${audit}
      else
         . clean.sh;sh run_loadrun.sh 12 ${mode} 28 mobilenet 112  AccuracyOnly ${audit}
      fi
   fi

   wait
fi

if [ "$performance" == "1" ];then

for((i=1;$i<=$itime;i=$(($i + 1))));  
do
   rm mlperf_log*
   if [ "${mode}" == "server" ];then
      if [ "${model}" == "resnet50" ];then
         . clean.sh;sh run_loadrun.sh 2 ${mode} 28 resnet50 112 PerformanceOnly ${audit}
      else
         . clean.sh;sh run_loadrun.sh 4 ${mode} 28 mobilenet 112  PerformanceOnly ${audit}
      fi
   else
      if [ "${model}" == "resnet50" ];then
         . clean.sh;sh run_loadrun.sh 16 ${mode} 28 resnet50 112 PerformanceOnly ${audit}
      else
         . clean.sh;sh run_loadrun.sh 12 ${mode} 28 mobilenet 112  PerformanceOnly ${audit}
      fi
   fi
   wait

   if [ "${mode}" == "server" ];then
      mode="Server"
   fi
   if [ "${mode}" == "singlestream" ];then
      mode="SingleStream"
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


run_mobilenet_server(){
if [ "$performance" == "1" ];then
for((i=1;$i<=$itime;i=$(($i + 1))));  
do
   rm mlperf_log*
   . clean.sh;sh run_loadrun.sh 4 server 28 mobilenet 112  PerformanceOnly $audit 
   wait
   mkdir -p results/mobilenet/Server/performance/run_${i}
   mv results/mobilenet/Server/performance/mlperf_log_* results/mobilenet/Server/performance/run_${i} 

   if [[ "$i" == "1"  && "${audit}" != "" ]];then
   mkdir -p audit/mobilenet/Server/${audit}/performance/run_1
   mv audit/mobilenet/Server/${audit}/performance/mlperf_log_* audit/mobilenet/Server/${audit}/performance/run_1/.
   fi

done
fi

if [ "$accuracy" == "1" ];then
   . clean.sh;sh run_loadrun.sh 4 server 28 mobilenet 112 AccuracyOnly $audit
   wait
fi
}

run_singlestream(){
model=$1

if [ "$performance" == "1" ];then
rm mlperf_log*
sh single_stream.sh ${model} PerformanceOnly $audit
wait
fi

if [ "$accuracy" == "1" ];then
rm mlperf_log*
sh single_stream.sh ${model} AccuracyOnly $audit
wait
fi
}

sed -i "s/EULER_SHARED_WORKSPACE=[0-1]/EULER_SHARED_WORKSPACE=0/" netrun.sh
sed -i "s/EULER_SHARED_WORKSPACE=[0-1]/EULER_SHARED_WORKSPACE=0/g" single_stream.sh

run_mlperf server resnet50
run_mlperf offline resnet50
run_mlperf server mobilenet
run_mlperf offline mobilenet
run_singlestream resnet50 
run_singlestream mobilenet 

