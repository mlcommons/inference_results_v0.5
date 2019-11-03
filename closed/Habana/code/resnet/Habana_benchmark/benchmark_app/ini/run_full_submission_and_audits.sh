audits=${1:-0}
preprocess=${2:-0} #pass second arg as 1 if you want to modify the data and preprocess it for audit test #3
if [ "$audits" = 'all' ]; then
   audits='1 3 4-A 5'
fi
echo $audits
devision='closed'
base_dir='./habana_submission0.5/'$devision'/Habana'
results='/results/Goya_1'
measurements='/measurements/Goya_1'
path_to_coco_val_data='dataset-coco-2017-val'
path_to_imagenet_val_data='dataset-imagenet-ilsvrc2012-val'
mlperf_accuracy_test='/home/ihubara/mlperf_inference/v0.5/classification_and_detection/tools/'
preprocess_path='
/home/ihubara/audit_habana_submission0.5/Release/habana_submission0.5/closed/Habana/code/resnet/Habana_benchmark/preprocess'
models_dir='/home/rshefer/projects/mlperf_models/'
image_dataset_base='/home/rshefer/projects/'
coco_dataset_base='/home/ihubara/coco'
scenarios='Offline SingleStream MultiStream Server'
benchmarks='resnet ssd-large'

if [ "$audits" = '0' ]; then
   modes='performance accuracy'
else
   modes='performance'
fi   
if [ "$audits" != '0' ] && [ "$preprocess" = '1' ]; then
   ##Modify the data and preprocess it for audit test #3
   cp '../audit/TEST03'$audit/modify* ./
   python3.6 modify_image_data.py -d $image_dataset_base/$path_to_imagenet_val_data -o $image_dataset_base/$path_to_imagenet_val_data'_test3' --dataset  imagenet
   cp $image_dataset_base/$path_to_imagenet_val_data/val* $image_dataset_base/$path_to_imagenet_val_data'_test3/'
   imagenet_file_break='0 1 2 3 4 5'
   for i in $imagenet_file_break; do
      mv $image_dataset_base/$path_to_imagenet_val_data'_test3/imagenet/ILSVRC2012_val_000'$i* $image_dataset_base/$path_to_imagenet_val_data'_test3/' 
   done
   python3.6 $preprocess_path/mlperf_resnet50_preprocess_and_compile.py False 50000 $image_dataset_base/$path_to_imagenet_val_data'_test3' './' resnet50_v1_batch10.recipe 10 'imagenet_habana_test3'
   python3.6 modify_image_data.py -d $coco_dataset_base -o $coco_dataset_base'_test3' --dataset  coco
   if [ -d "$coco_dataset_base'_test3'" ]; then rm -Rf $coco_dataset_base'_test3'; fi
   mv $coco_dataset_base'_test3'/coco/* $coco_dataset_base'_test3'
   python3.6 $preprocess_path/mlperf_ssd-resnet34_preprocess_and_compile.py -pr -ic 5000 -ip $coco_dataset_base'_test3' -m './' -hr ssd_resnet_recipe_500_b16.recipe -b 1 -prd 'coco_habana_test3'
   mv $coco_dataset_base'_habana_test3' $image_dataset_base
fi   
mkdir --parents  $base_dir'/systems'
#cp Goya_1.json $base_dir'/systems/'
cp /home/ihubara/old_submission_infernce_0_5/habana_submission0.5/closed/Habana/systems/Goya.json $base_dir'/systems/Goya_1.json'

for audit in $audits; do
   for bm in $benchmarks; do
      mkdir --parents $base_dir/$measurements'/'$bm
      for sc in $scenarios; do
         if [ "$audit" != '0' ]; then
            echo 'running test audit #'$audit' for '$bm' '$sc
            cp '../audit/TEST0'$audit/verify* ./
            cp '../audit/TEST0'$audit/audit* ./
         fi
         mkdir --parents $base_dir/$measurements'/'$bm'/'$sc
         python3.6 write_system_desc_id_imp.py -o $base_dir/$measurements'/'$bm'/'$sc'/Goya_1_'$sc'.json' -swf 'onnx-model from zenado'
         if [ "$audit" = '0' ] || [ "$audit" = '3' ]; then
            modes='performance accuracy'
         else
            modes='performance'
         fi 
         for mode in $modes; do
            mkdir --parents $base_dir$results'/'$bm'/'$sc'/'$mode
            if [ "$mode" = "accuracy" ]; then
               if [ "$audit" != '3' ]; then
                  echo 'Running accuracy '$bm' '$sc
                  ./HABANA_benchmark_app $bm'_'$sc'_accuracy.ini'
                  if [ "$bm" = "resnet" ]; then
                     python3.6 $mlperf_accuracy_test/accuracy-imagenet.py --mlperf-accuracy-file ./mlperf_log_accuracy.json --imagenet-val-file $image_dataset_base/$path_to_imagenet_val_data/val_map.txt | tee accuracy.txt
                  else
                     python3.6 $mlperf_accuracy_test/accuracy-coco.py --mlperf-accuracy-file ./mlperf_log_accuracy.json --coco-dir $image_dataset_base/$path_to_coco_val_data --remove-48-empty-images | tee accuracy.txt
                  fi
                  mv accuracy.txt $base_dir/$results'/'$bm'/'$sc'/'$mode
                  mv mlperf_log* $base_dir/$results'/'$bm'/'$sc'/'$mode
                  rm $base_dir/$results'/'$bm'/'$sc'/'$mode'/mlperf_log_trace.json'
                  cp user* $base_dir/$measurements'/'$bm'/'$sc
                  cp mlperf.conf $base_dir/$measurements'/'$bm'/'$sc
                  touch $base_dir/$measurements'/'$bm'/'$sc'/README.md'
               else
                  python change_data_test3.py -i $bm'_'$sc'_accuracy.ini'
                  echo 'Running audit TEST03 accuracy '$bm' '$sc
                  ./HABANA_benchmark_app $bm'_'$sc'_accuracy.ini'
                  python change_data_test3.py -i $bm'_'$sc'_accuracy.ini'
                  if [ "$bm" = "resnet" ]; then
                     python3.6 $mlperf_accuracy_test/accuracy-imagenet.py --mlperf-accuracy-file ./mlperf_log_accuracy.json --imagenet-val-file $image_dataset_base/$path_to_imagenet_val_data/val_map.txt | tee accuracy.txt
                  else
                     python3.6 $mlperf_accuracy_test/accuracy-coco.py --mlperf-accuracy-file ./mlperf_log_accuracy.json --coco-dir $image_dataset_base/$path_to_coco_val_data --remove-48-empty-images | tee accuracy.txt
                  fi
                  mv accuracy.txt $base_dir'/audit/'$bm'/'$sc'/TEST0'$audit'/accuracy'
                  mv mlperf_log* $base_dir'/audit/'$bm'/'$sc'/TEST0'$audit'/accuracy'
                  rm  $base_dir'/audit/'$bm'/'$sc'/TEST0'$audit'/accuracy/mlperf_log_trace.json'
               fi   
            else
               if [ "$sc" = "Server" ] && [ "$audit" = '0' ]; then
                  runs='1 2 3 4 5'
               else
                  runs='1'
               fi
               for r in $runs; do
                  echo 'Running performance '$bm' '$sc' run '$r
                  if [ "$audit" = '3' ]; then
                     echo 'Modifying data'
                     python change_data_test3.py -i $bm'_'$sc'_performance.ini'
                     ./HABANA_benchmark_app $bm'_'$sc'_performance.ini'
                     python change_data_test3.py -i $bm'_'$sc'_performance.ini'
                  else
                     ./HABANA_benchmark_app $bm'_'$sc'_performance.ini'
                  fi   
                  if [ "$audit" = '0' ]; then
                     mkdir --parents $base_dir/$results'/'$bm'/'$sc'/'$mode'/run_'$r
                     #python3.6 write_system_desc_id_imp.py -o $base_dir/$measurements'/'$bm'/'$sc'/system_desc_id_imp.json' -swf 'onnx-model from zenado'
                     mv mlperf_log* $base_dir/$results'/'$bm'/'$sc'/'$mode'/run_'$r
                     rm $base_dir/$results'/'$bm'/'$sc'/'$mode'/run_'$r'/mlperf_log_trace.json'
                  else
                     echo 'Running Audit '$sc' scenario run '$r
                     mkdir -p $base_dir'/audit/'$bm'/'$sc'/TEST0'$audit'/accuracy'
                     mkdir -p $base_dir'/audit/'$bm'/'$sc'/TEST0'$audit'/performance/run_'$r
   
                     if [ $r -eq 1 ]; then
                        if [ "$audit" = '1' ]; then
                           python3.6 verify_accuracy.py -a $base_dir$results'/'$bm'/'$sc'/accuracy/mlperf_log_accuracy.json' -p ./mlperf_log_accuracy.json | tee verify_accuracy.txt
                        fi
                        if [ "$audit" != '4-A' ]; then
                           python3.6 verify_performance.py -r $base_dir$results'/'$bm'/'$sc'/performance/run_'$r'/mlperf_log_summary.txt' -t ./mlperf_log_summary.txt | tee verify_performance.txt
                           cp mlperf_log_acc* $base_dir'/audit/'$bm'/'$sc'/TEST0'$audit'/accuracy'
                           cp mlperf_log* $base_dir'/audit/'$bm'/'$sc'/TEST0'$audit'/performance/run_'$r
                        else  
                           cp mlperf_log_acc* $base_dir'/audit/'$bm'/'$sc'/TEST0'$audit'/accuracy'
                           cp mlperf_log* $base_dir'/audit/'$bm'/'$sc'/TEST0'$audit'/performance/run_'$r
                           cp ../audit/TEST04-B/audit* ./
                           mkdir -p $base_dir'/audit/'$bm'/'$sc'/TEST04-B/accuracy'
                           mkdir -p $base_dir'/audit/'$bm'/'$sc'/TEST04-B/performance/run_'$r
                           echo 'Running performance'$bm' '$sc' run '$r' Test 4-B'
                           ./HABANA_benchmark_app $bm'_'$sc'_performance.ini' 
                           mv mlperf_log_acc* $base_dir'/audit/'$bm'/'$sc'/TEST04-B/accuracy'
                           mv mlperf_log* $base_dir'/audit/'$bm'/'$sc'/TEST04-B/performance/run_'$r
                           rm $base_dir'/audit/'$bm'/'$sc'/TEST04-B/performance/run_'$r'/mlperf_log_trace.json'
                           echo 'Running Test04 verify_test4_performance'
                           python3.6 verify_test4_performance.py -u $base_dir'/audit/'$bm'/'$sc'/TEST0'$audit'/performance/run_'$r'/mlperf_log_summary.txt' -s $base_dir'/audit/'$bm'/'$sc'/TEST04-B/performance/run_'$r'/mlperf_log_summary.txt' | tee verify_performance.txt
                        fi
                     fi
                     if [ "$audit" = '1' ]; then
                        mv verify_accuracy.txt $base_dir'/audit/'$bm'/'$sc'/TEST0'$audit
                     fi   
                     mv verify_performance.txt $base_dir'/audit/'$bm'/'$sc'/TEST0'$audit
                  fi
               done
            fi   
         done
      done
      echo 'Done '$bm
   done
   if [ "$audit" != '0' ]; then
      rm modify_image_data*
   fi   
   if [ "$audit" != '0' ]; then
      rm verify_*
      rm audit.config
      echo 'removed audit files of '$audit
   fi
done   
echo All Done
   