Follow the MLPerf instructions to build LoadGen, etc.

$ pip install tensorflow-1.14.0rc0-cp37-cp37m-linux_x86_64.whl

$ python python/main.py     --backend=tflite-ncore-offline-imagenet     --cache=1     --config=../mlperf.conf     --dataset-path=/n_mounts/scr_ncore/parvizp/Git/CK-TOOLS/dataset-imagenet-ilsvrc2012-val     --max-batchsize=128     --model=models/mobilenet_v1_1.0_224.tflite     --model-name=mobilenet     --profile=mobilenet-tf-ncore-offline     --scenario=Offline     --threads=2      --output=/tmp/mlperf-tempout/output

$ python python/main.py     --backend=tflite-ncore-offline-imagenet     --cache=1     --config=../mlperf.conf     --dataset-path=/n_mounts/scr_ncore/parvizp/Git/CK-TOOLS/dataset-imagenet-ilsvrc2012-val     --max-batchsize=128     --model=models/mobilenet_v1_1.0_224.tflite     --model-name=mobilenet     --profile=mobilenet-tf-ncore-offline     --scenario=Offline     --threads=2      --accuracy --output=/tmp/mlperf-tempout/output


