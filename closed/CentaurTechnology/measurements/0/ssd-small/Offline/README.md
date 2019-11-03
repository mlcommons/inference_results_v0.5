Follow the MLPerf instructions to build LoadGen, etc.

$ pip install tensorflow-1.14.0rc0-cp37-cp37m-linux_x86_64.whl

$ python python/main.py     --backend=tflite-ncore     --cache=1     --config=../mlperf.conf     --dataset-path=/n_mounts/scr_ncore/parvizp/Git/CK-TOOLS/dataset-coco-2017-val     --max-batchsize=1     --model=models/ssd_mobilenet_v1_coco_2018_01_28.tflite     --model-name=ssd-mobilenet     --profile=ssd-mobilenet-tf-ncore     --scenario=Offline     --threads=2      --output=/tmp/mlperf-tempout/output

$ python python/main.py     --backend=tflite-ncore     --cache=1     --config=../mlperf.conf     --dataset-path=/n_mounts/scr_ncore/parvizp/Git/CK-TOOLS/dataset-coco-2017-val     --max-batchsize=1     --model=models/ssd_mobilenet_v1_coco_2018_01_28.tflite     --model-name=ssd-mobilenet     --profile=ssd-mobilenet-tf-ncore     --scenario=Offline     --threads=2      --accuracy --output=/tmp/mlperf-tempout/output


