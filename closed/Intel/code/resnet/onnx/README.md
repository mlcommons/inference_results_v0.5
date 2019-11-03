# NNP-I MLPerf LoadRunner

This repo includes mlperf benchmark load runner for spring hill. It uses MLPerf's loadgen for load generation.

## Preparations:
### Prequiesities:
Install the following dependencies:
```
sudo yum install opencv opencv-devel cmake glib2-devel
pip install numpy absl-py
```
### Datasets
- Go to [this link](https://github.com/ctuning/ck) and download collective knowledge.
  You can validate the download using this command:
  ```
  $ ck version
  V1.10.3
  ```
  Next, pull it's repositories containing dataset packages:
  ```
  $ ck pull repo:ck-env
  ```
- Download your desired dataset.
  For imagenet for example run:
  ```
  $ ck install package --tags=image-classification,dataset,imagenet,val,original,full
  $ ck install package --tags=image-classification,dataset,imagenet,aux
  ```
  and then copy the labels next to the images:
  ```
  $ ck locate env --tags=image-classification,dataset,imagenet,val,original,full
  /home/user/CK-TOOLS/dataset-imagenet-ilsvrc2012-val
  $ ck locate env --tags=image-classification,dataset,imagenet,aux
  /home/user/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux
  $ cp `ck locate env --tags=aux`/val.txt `ck locate env --tags=val`/val_map.txt
  ```
### Model
- Get a compiled blob of your desired model
- Prepare an ONNX model and use the Intel NNP-I Graph Transformer to create a blob.
### Get NNP-I full package
- Get the full NNP-I Graph Transformer (at least version 0.3.1) and create a blob from the ONNX model.
- Make the sph card is up and running:
	```
	$ sudo sph_stat_host -n
	```

## Compiling Loadrunner:
- Clone this repository.
- Run the following commands:
	```
	$ mkdir cmake-build && cd cmake-build
	$ cmake ..
	$ make
	```

## Running example:
```
./single_stream.sh -i ~/CK-TOOLS/dataset-imagenet-ilsvrc2012-val/ -c ~/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt -b ~/models/prepared_blobs/resnet50.zip --num_devices 1 --num_dev_nets_per_device 1 --num_parallel_infers 3

```
This will run the test of single stream scenario on your compiled blob. The output files will be written under `output_dir/single_stream`.
Note that in single stream scenario the measured metric is latency.
There are corresponding scripts for each scenario with the same usage, simply run:
``` ./<script_name> -h ``` for help.

### Usage:
```
Usage: ./<script_name> [-i|--input_data <path to input data>] [-c|--class_file <path to classification file>] [-b|--blob_file <path to the compiled blob>] [-o|--output_dir] [--num_devices] [--num_dev_nets_per_device] [--num_parallel_infers]
-i|--input_data               (REQUIRED) path to input data
-c|--class_file               (REQUIRED) files to check classification res
-b|--blob_file                (REQUIRED) path to the compiled blob
-o|--output_dir               (OPTIONAL) path to output directory, default is ./output_dir
--num_devices                 (OPTIONAL) number of devices to use, default is 1
--num_dev_nets_per_device     (OPTIONAL) number of device networks to create per device, default is 12
--num_parallel_infers         (OPTIONAL) number of infers to use on every device network, default is 2
-h|--help                     Display this help message
```
