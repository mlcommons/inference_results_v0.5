
# Inference BKM for PyTorch (MLPerf v0.5 compatible)

# OS
We tested it on CentOS only.

# Prepare

·        Python  >=3.6,

·        GCC 7

·        ICC (Intel C/C++ Compiler) 19 version 
         from Intel Parallel Studio XE https://software.intel.com/en-us/c-compilers




# Download and build PyTorch

 Create root folder

 mkdir mlperf_submit

 cd mlperf_submit

 Download and build pytorch for int8 model generation

 Pip install pyyaml numpy

 git clone https://github.com/pytorch/pytorch.git

 cd pytorch

 git fetch origin pull/25235/head:mlperf

 git checkout mlperf


# Generate int8 model
 

git submodule update --init --recursive

export MKLROOT=/home/huiwu1/src/mkl2019_3

export USE_OPENMP=ON

export CAFFE2_USE_MKLDNN=ON

export MKLDNN_USE_CBLAS=ON

python3.6 setup.py build

cd ..

Please follow this to generate int8 model.

https://github.com/intel/optimized-models/tree/v1.0.9/pytorch/mlperf_tools

 

# Build Deep-learning-math-kernel

\#make sure to install icc19

"Sudo apt-get install libgflags-dev" on Unbuntu  or "sudo yum install gflags-devel" on Centos

 

export BOOST_ROOT=<boost_install_folder>/include

cd /path/to/mlperf_submit/pytorch

cd third_party/ideep/euler/

mkdir build; cd build

cmake3 .. -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DWITH_VNNI=2

make 

# Build pytorch with Deep-learning-math-kernel

cd /path/to/mlperf_submit/pytorch

git submodule update --init --recursive

export MKLROOT=/home/huiwu1/src/mkl2019_3

export USE_OPENMP=ON

export CAFFE2_USE_MKLDNN=ON

export MKLDNN_USE_CBLAS=ON

export CAFFE2_USE_EULER=ON

python3.6 setup.py build

cd ..

# Get and build mlperf scripts

cd /path/to/mlperf_submit/pytorch 

cp -r <Submission_Package>/code/resnet/pytorch-caffe2 mlperf-inference-loadgen-app-cpp

cd mlperf-inference-loadgen-app-cpp

mkdir models; mkdir models/resnet50; mkdir models/mobilenet

cp /path/to/mlperf_submit/optimized-models/pytorch/mlperf_tools/inference/models/resnet50/init_net_int8.pb ./models/resnet50/init_net_int8_euler.pb

cp /path/to/mlperf_submit/optimized-models/pytorch/mlperf_tools/inference/models/resnet50/predict_net_int8_small_bs.pbtxt ./models/resnet50/predict_net_int8_euler_lat.pbtxt

cp /path/to/mlperf_submit/optimized-models/pytorch/mlperf_tools/inference/models/resnet50/predict_net_int8_large_bs.pbtxt ./models/resnet50/predict_net_int8_euler.pbtxt

 

cp /path/to/mlperf_submit/optimized-models/pytorch/mlperf_tools/inference/models/mobilenet/init_net_int8.pb ./models/mobilenet/init_net_int8_euler.pb

cp /path/to/mlperf_submit/optimized-models/pytorch/mlperf_tools/inference/models/mobilenet/predict_net_int8_small_bs.pbtxt ./models/mobilenet/predict_net_int8_euler_lat.pbtxt

cp /path/to/mlperf_submit/optimized-models/pytorch/mlperf_tools/inference/models/mobilenet/predict_net_int8_large_bs.pbtxt ./models/mobilenet/predict_net_int8_euler.pbtxt

cd ..

  

# For roofline test

cd mlperf-inference-loadgen-app-cpp/scripts

export MKLROOT=/path_to_mkl_dnn/mkl2019_3

make clean; make

 

Get the imagenet2012 dataset and label file following https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection -> DataSets. 

cp <image_net_2012_label_file>/val.txt ./val.txt

 
# Build Pytorch Backend

export MKLROOT=/path_to_mkl_dnn/mkl2019_3

cd <mlperf_submit>/pytorch/mlperf-inference-loadgen-app-cpp/scripts/backend

make clean

make

# Setup MLPerf Loadgen

Download loadgen and build as a library. This BKM is created with version 55c0ea4e772634107f3e67a6d0da61e6a2ca390d.For submision, use the one for 0.5 submission from Mlperf.

#ICC instruction for CentOS

source      /opt/intel/compilers_and_libraries_2019/linux/bin/compilervars.sh intel64 

cd /path/to/mlperf_submit/

pip install absl-py numpy

git clone --recurse-submodules https://github.com/mlperf/inference.git <mlperf_submit>/pytorch/third_party/mlperf_inference

cd <mlperf_submit>/pytorch/third_party/mlperf_inference/loadgen

git checkout 55c0ea4e772634107f3e67a6d0da61e6a2ca390d

git apply <mlperf_submit>/pytorch/mlperf-inference-loadgen-app-cpp/mlperf_inference.patch 

CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel

pip install --force-reinstall dist/mlperf_loadgen-*.whl



cd build

cmake3 .. && cmake3 --build .

cp libmlperf_loadgen.a ../

cp libmlperf_loadgen.a ../../



export MKLROOT=/path_to_mkl_dnn/mkl2019_3

cd <mlperf_submit>/pytorch/mlperf-inference-loadgen-app-cpp/scripts/backend

make clean

make

cd ../..



Modify path in <mlperf_submit>/pytorch/mlperf-inference-loadgen-app-cpp/loadrun/Makefile and *.sh 

For Makefile, it may be needed to change following path to the actual location in your system,

`CAFFE2_DIR - the pytorch code path`  
`LOADGEN_DIR - path to Mlperf inference project`  
`BOOST_DIR - path to boost library`   

For below scripts,

`single_stream.sh`

`netrun.sh`

`loadrun.sh`

Change following path to the actual location in your system if needed,

`CAFFE2_DIR - the pytorch code path`

`LD_PRELOAD=/opt/intel/compilers_and_libraries/linux/lib/intel64/libiomp5.so`

`--images "/lustre/dataset/imagenet/img/ILSVRC2012_img_val/"`  

`--labels "val.txt"`  

`--init_net_path ../models/resnet50/init_net_int8.pb`  

`--predict_net_path ../models/resnet50/predict_net_int8.pbtxt` 



cd loadrun

make clean

make

 

cp <image_net_2012_label_file>/val.txt ./val.txt

cp <mlperf_submit>/pytorch/third_party/mlperf_inference/v0.5/mlperf.conf ./



# Run MLPerf Loadgen Test

**Single Stream scenario**  
loadgen and loadrun are built in one executable for single stream scenario.  

./single_stream.sh resnet50 PerformanceOnly

`resnet50  - model to test resnet50|mobilenet` 

`PerformanceOnly - loadgen mode(PerformanceOnly|AccuracyOnly)`

Following will be displayed after execution.    

`Completed` 

Check **mlperf_log_summary.txt** for MLPerf Loadgen benchmark result.   

You may update the OMP_NUM_THREADS and “numactl -C” for your platform in single_stream.sh to run test on 1 socket.

export OMP_NUM_THREADS=28 

numactl -C 0-27 …

**Offline scenario**   

To run test with 14 instance on 56 cores:

./run_loadrun.sh 11 offline 14 resnet50 56 PerformanceOnly

It launches 1 loadrun and 14 netrun instances and executes offline scenario.  
`11 - batch size`

`Offline - Loadgen scenario (offline|server|singlestream)`  

`14 - number of instances`

`resnet50 - model to test resnet50|mobilenet`

`56 - number of system processors`

`PerformanceOnly - loadgen mode(PerformanceOnly|AccuracyOnly)`

After the execution is done, following will be displayed.  
`Completed`  

Check mlperf_log_summary.txt for result from loadgen. It may be moved to  ./results/offline/resnet50/ 

And you may change the instance number (lines of netrun.sh) in m_14_56_shm.sh and update the environments in netrun.sh to run different core and instance configurations. 

**Server scenario**    

./run_loadrun.sh 2 server 14 resnet50 56 PerformanceOnly

**Measure Accuracy**

Get MLPerf accuracy
script from https://github.com/mlperf/inference/blob/master/v0.5/classification_and_detection/tools/accuracy-imagenet.py

Extract validation label file from imagenet data set, copy it to ./val_map.txt

Change PerformanceOnly to AccuracyOnly, e.g.,

`./single_stream.sh resnet50 AccuracyOnly`

After execution is done, play below cmd,

`python3 accuracy-imagenet.py --mlperf-accuracy-file  ./results/singlestream/resnet50/mlperf_log_accuracy.json --imagenet-val-file val_map.txt --dtype int32`

Output is like following,

`accuracy=76.146%, good=38073, total=50000`

**Mlperf.conf and User.conf**

The content of Mlperf.conf cannot be changed, which is defined by Mlperf rules.

The User.conf can be modified to specify settings, following 2 are mostly useful,

`*.Server.target_qps = 2300 - change it lower to get lower server latency`

`*.Offline.target_qps = 5000 - change it higher to have Mlperf loadgen to generate more queries`

**Run test on platform with other configurations**

Offline scenario:  

./run_loadrun.sh 11 offline 14 resnet50 56 PerformanceOnly

Server scenario:    

./run_loadrun.sh 2 server 14 resnet50 56 PerformanceOnly

# FAQ

**Prepare Boost Interprocess Library**  

If build error reports missing Boost Interprocess Lib header files.

Refer https://www.boost.org/users/history/version_1_70_0.html It's a header only library.  

**Segmentation fault**

 It may be caused by different components built by different version of compilers. Try build both porytch and backends with GCC 7.x.

**Bus error**

Often cause by no sufficient shared memory can be allocated. Use df -h to check /dev/shm.
