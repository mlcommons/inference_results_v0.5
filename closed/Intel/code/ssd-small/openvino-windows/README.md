# Instruction on using OpenVino Software for Windows


## Prerequisites
+ Windows version: 10
+ Microsoft Visual Studio: 15 2017 (with MSBuild)
+ [cmake](https://cmake.org/download/): 3.15.4 (Download Windows Installer, Make sure to select option to add cmake to PATH)
+ [Boost](https://www.boost.org/users/history/version_1_70_0.html): 1.71.0
+ [Intel C++ Compiler 19.0](https://software.intel.com/en-us/intel-parallel-studio-xe) (During installation, check that the VS 2017 is detected - the installation interface will show you any unmet dependencies)
+ [wget](http://gnuwin32.sourceforge.net/downlinks/wget.php) (Add the wget bin folder to PATH)


## Models

#### SSD-MobileNet
See calibration/OpenVino_calibration.md#quantized-model-example

#### ResNet-50

See calibration/OV_RN-50-sample/RN-50-calibration.md

#### MobileNet
See calibration/OV_MobileNet-sample/MobileNet-calibration.md


## Setup Instructions


### Loadgen

Please use commit hash 413dbabcb30dc2ee1fe42e7b8090b37e8144617d and apply PR 482, 489, and 502

```
pip install absl-py
git clone --recurse-submodules https://github.com/mlperf/inference.git mlperf_inference
cd mlperf_inference\loadgen
git checkout 413dbabcb30dc2ee1fe42e7b8090b37e8144617d
git pull origin pull/482/head
git pull origin pull/489/head
git pull origin pull/502/head

mkdir build && cd build
cmake ..
cmake --build . --config Release

copy Release/mlperf_loadgen.lib ..\
```

### OpenVINO installation
Build from [dldt](https://github.com/opencv/dldt). Follow the build instructions for [Windows](https://github.com/opencv/dldt/tree/master/inference-engine#build-on-windows-systems).

Add the following flags when generating the solution: ``` -DTHREADING=OMP -DCMAKE_BUILD_TYPE=Release -DENABLE_CLDNN=ON -DENABLE_OPENCV=ON DENABLE_MKL_DNN=ON -DENABLE_DLIA=OFF -DENABLE_VPU=OFF -DENABPP=OFF -DENABLE_PROFILING_ITT=OFF -DENABLE_VALIDATION_SET=OFF -DENABLE_TESTS=OFF -DENABLE_GNA=OFF```


### Build source code
```
cd ov_mlperf
mkdir build && cd build

set "InferenceEngine_DIR=<\path\to\dldt>\inference-engine\build"
set "OpenCV_DIR=<\path\to\dldt\opencv>\cmake"

cmake -G "Visual Studio 15 2017 Win64" ^
	-DLOADGEN_DIR=<\path\to\mlperf_inference>\loadgen ^
	-DBOOST_INCLUDE_DIRS=\path\to\boost_1_71_0 ^
	-DIE_EXTENSION_LIB=<\path\to\dldt>\inference-engine\bin\intel64\Release\cpu_extension.lib ^
	-DIE_LIBRARY=<\path\to\dldt>\inference-engine\bin\intel64\Release\inference_engine.lib ^
	-DIE_SRC_DIR=<\path\to\dldt>\inference-engine\src ^
	-DCMAKE_BUILD_TYPE=Release ..

cmake --build . --config Release
```

**NB**: The above step builds the libraries and plugins necessary for running openvino on CPU or GPU.
To run inference on CPU+GPU the MultiDevicePlugin is required (which is not part of the dldt).
If you want to run in this 'MULTI' mode, you have to install the OpenVINO distribution R3 (or above), and copy the following files:
+ MultiDevicePlugin.dll, MultiDevicePlugin.lib, plugins.xml

**from** "C:\Program Files (x86)\IntelSWTools\openvino\inference_engine\bin\intel64\Release\" **into** <\path\to\dldt>\inference-engine\bin\intel64\Release

## Running Workloads

Batch scripts for the 6 measurements are provided in scripts folder. To run:

* Edit the relevant entries in ```<*>``` as appropriate (especially, path to models, mlperf and user configs).

* Add relevant libraries to User PATH variable:
    * OpenCV and InferenceEngine dll paths:

```
set "PATH=<\path\to\dldt>\inference-engine\bin\intel64\Release;%PATH%"
set "PATH=<\path\to\dldt\opencv>bin;%PATH%"
```

* Run Edited script:

```
int8_cpu_mobilenet_single.bat
```




