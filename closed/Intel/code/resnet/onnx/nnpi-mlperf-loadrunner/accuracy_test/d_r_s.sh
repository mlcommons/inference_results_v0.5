cd ..
source short_accuracy_test.sh --blob /home/user/new_mlperf/models/resnet50/resnet50_gemmlowp_alpha.zip -i /home/user/CK-TOOLS/dataset-imagenet-ilsvrc2012-val/ -c /home/user/new_mlperf/data/imagenet/label/val.txt
cd accuracy_test
