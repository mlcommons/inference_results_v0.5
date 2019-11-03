The code enables "Single Stream", "Offline", "Server", "MultiStream" scenarios for resnet50 and mobilenet benchmarks. ssd-mobilenet, ssd-resnet34 benchmarks enabling is in progress.

Dependencies:
- install OpenCV [instructions](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)

Build instructions:
- Building OpenVino with OpenMP:
  git clone https://github.com/opencv/dldt.git

  cd ./dldt
  
  git checkout pre-release
  
  git submodule update --init --recursive

  cd ./inference-engine
  
  mkdir build && cd build
  
  cmake -DCMAKE_BUILD_TYPE=Release -DTHREADING=OMP -DENABLE_DLIA=OFF -DENABLE_VPU=OFF -DENABPP=OFF -DENABLE_PROFILING_ITT=OFF -DENABLE_VALIDATION_SET=OFF -DENABLE_TESTS=OFF -DENABLE_GNA=OFF -DENABLE_CLDNN=OFF ..
  
  make MKLDNNPlugin -j 224
  
  cd ../bin/intel64/Release/
  
  Please note the library paths in /bin/intel64/Release/

- Build MLPerf Loadgen cpp library (Follow instructions on MLPerf github)

- Replace the install paths in CMakeLists.txt and also include the following

    export InferenceEngine_DIR=/path/to/dldt/inference-engine/build

- Now build ov_mlperf application

    mkdir build && cd build

    cmake ..
    
    make
    
Running SingleStream/offline/Server/MultiStream scenarios:
- Run the corresponding scripts under scripts/ folder

