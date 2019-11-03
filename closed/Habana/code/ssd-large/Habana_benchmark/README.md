***************************************************************************************
********************************** 1. MLperf Habana benchmark ************************
***************************************************************************************

the benchmark can be compiled and build using one of two build environments:
1. GUI interface using VSCODE and CMAKE-GUI. These options allow easy debugging and code view using vscode
2. Command line build using - build.sh

To run the MLperf benchmark the user must run an image and model pre-processing phase using the supplied python scripts
that are under the preprocess directory



***************************************************************************************
************************************ 2. Environment *********************************
***************************************************************************************
the Habana MLPERF bechmark was tested with the following HW and SW configuration.

(1) HW 
    (1) 2 sockets
    (2) 10 cores per socket - Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz
    (3) Host memory - 64GB of RAM
    (4) Single Goya card - 16GB DDR

(2)
    (1) Tested on Ubuntu 16.04
    (2) Habana SW tools package V0.2.0
    (3) GCC-5.4 or CLANG++-8 - inatsalled CLANG is a precodnition to build the MLPERF loadgen library
    (4) Python 3.6
    (5) Onnx package - 1.4.1
    (6) Numpy python package - 1.14.1
    (7) Habana tool set - V0.2.0
    (8) TF -    1.11.0
    (9) MXNET - 1.3.1
    (10) tourchvision - 0.2.1
    (11) PIL - 5.2.0



***************************************************************************************
************************************** 3. preconditions *******************************
***************************************************************************************
user should verify proper installation of the following SW packages:
1. cmake V3.5.1 or above, if user wishes to use the GUI interface cmake-gui should be installed
2. CLANG8 - installation 
3. VSCODE this is for a user who wishes to use the vscode GUI for debug and build otherwise
   this installation could be skipped.
   the following VSCODE should be installed V1.3 including the following VSCODE plugins:
   a) c/c++ 0.24.1
   b) c++ intellisense 0.2.2
   c) Cmake Tools 1.1.3
   d) CMake 0.0.17
   e) CMake integration 0.6.1
   f) Cmake Tools Helper 0.2.1


***************************************************************************************
********************************* 4. Environment variables ****************************
***************************************************************************************
The following environment variables must be defined before beginning the build process, 

1. export MLPERF_LOADGEN_PATH=/home/labuser/MLperf/inference/loadgen/
   --- path to loadgen source and header files directory
2. export MLPERF_BUILD_DIR=/home/labuser/Mlperf_build/
   --- path to the build directory (where all intermediate binary files will be generated)
3. export MLPERF_MODEL_DIR=/home/labuser/mlperf_models/
   --- path to onnx and recipe file directory (it is assumed that use run the compilation
       script before building the benchmark)
4. export MLPERF_LIB_PATH=/home/labuser/MLperf/inference/out/MakefileGnProj/obj/loadgen/
   --- path to the loadgen static library (it is assumed that the user builds the loadgen library before building 
       the Habana benchmark)
5. export MLPREF_BENCHMARK_CODE_DIR = $MLPERF_BASE_PATH/demos/Habana_benchmark
   --- path to the directory containing the benchmark code directory 
6. export MLPERF_INSTALL_DIR=$MLPERF_BASE_PATH/habana_benchmark_install/
   --- path to the installation directory (where the run directories will be placed)
   Note that the example above are should be treated as machine specific example and could vary according to the user
   directory arrangement.

   All these export command for the environment variables were placed in a bash script define_env.sh, user can change the script as needed and run it using the following command:

   source define_env.sh

   The environment variables must be set before any other build or run operation takes place

***************************************************************************************
******************** 5. Habana benchmark directory structure **************************
***************************************************************************************
[(1) Habana_benchmark]
        [(2) benchmark_app]
                [(3) Build]
                [(4) headers]
                [(5) ini]
                [(6) source]
                (7) CMakeLists.txt
        [(8) global_headers]
        [(9) runner]
                [(10) headers]
                [(11) source]
                (12) CMakeLists.txt
        [(13) utils]
        (14) CMakeLists.txt
        (15) README.md

(1) - root directory of the benchmark code.
(2) - directory contains the executable headers, source code and all other needed files to build and run the benchmark application
(3) - contains to files build.sh and rebuild.sh used to build the application.
(4) - all plication level header files
(5) - folder contains:
      (A) all ini files (for resnet50 and SSD)
      (B) all mlperf config files (mlperf.config, user.conf and audit.conf) 
(6) - all application source files.
(7) - application cmake main file for generating the application makefile
(8) - global_headers - these header files are been used by the benchmark application and also by the runner library.
(9) - runner - compiled into a static library that implements all needed scenarios (single, multi, server and offline) and
      interacts with loadgen static library (the runner implements all the needed callbacks requested by the loadgen library)
(10)- runner library header files
(11)- runner library source file
(12)- cmake file for building the runner makefile 
(13)- directory containing the preprocess (recipe compile and image preprocessing) python scripts

***************************************************************************************
************************************** 6. Building ************************************
***************************************************************************************
#1# Building using Command line
    *** Generate the build folder the user may place the build directory in any desired location
        as long as this location is aligned with the environment variables defined in stage 4.
    *** update the define_env.sh so MLPERF_BUILD_DIR points to the build directory location.
    *** run the following command ==> source define_env.sh
    *** To start the build process user needs to run build.sh script (placed under benchamr_app) this can be done in one of two options
          1. copy build.sh and revuild.sh to the build directory (for build.sh and rebuild.sh location please check section 6)
          2. run the build.sh
                (*) ./build.sh Release 
                    this build Release configurations
                (**) ./build.sh Debug
                    this build Debug configuration 
        the build script will generate a run directory under $MLPERF_INSTALL_DIR/app/Release or $MLPERF_INSTALL_DIR/app/Debug
        this file contains all needed files to run the benchmark application.

#2# building using vscode
     *** go through the steps of #1# Building using Command line, need to do for the first time as the build.sh script
         run the CMAKE scripts to generate the make file (Debug/Release) that are been used by the VSCODE task.json, an existing makefile are precondition for proper build task in the VSCODE environment.
     *** Build the application
          1. open vscode
          2. from the "file" menu, "open folder" and choose Habana_benchmark folder
          3. from terminal menu choose "run task" and choose the appropriate build task
               "build debug", "build release", "rebuild debug", "rebuild release", "clean debug", "clean Release"
          4. the Benchmark app will be generated under Habana_benchmark/benchmark_app/app/<Debug/Release>

         the vscode enables debugging all needed json files are part of this package.


***************************************************************************************
******************* 7. Image dataset preprocess and Recipe preparations ****************
***************************************************************************************

there are two major tasks to be handled during the preprocessing
1. preparation of image dataset (resize, crop and quantization), this process can be run for ImageNet or coco
2. preparing a recipe file from a model file (onnx, mxnet, pytorch or TF)

all the python scripts were placed under preprocess folder the following two main scripts are available:
    1. mlperf_resnet50_preprocess_and_compile.py - used to compiler resnet50_v1.onnx model to a recipe file and preprocess ImageNet test
       dataset.
    2. mlperf_ssd-resnet34_preprocess_and_compile.py used to compile a SSD recipe file or preprocess COCO image test data set


(1) Script usage for resnt50
    1. This option compiles a recipe file to be used by the Habana HW, the parameters to be used by the scrips are shown below.
        
        param[1] - True
        param[2] - Model directory full path
        param[3] - Onnx model file name
        param[4] - Calib image list file name
        param[5] - Source image directory full path
        param[6] - Batch size

    2. preprocessing ImageNet/SSD test image data base

        param[1] - False
        param[2] - Number of images to be preprocessed
        param[3] - Source image directory name full path
        param[4] - Model directory full path
        param[5] - Habana recipe file name
        param[6] - Batch size

(2) Script usage for resnt50
    TBD
***************************************************************************************
***************************** 8. running the benchmark ********************************
***************************************************************************************
The run environment is generated under the following location:
1. Release - $MLPERF_INSTALL_DIR/app/Release
2. Debug - $MLPERF_INSTALL_DIR/app/Debug

each folder above contains all the needed files to run the benchmark application including:
1. INI files
    there are three basic INI file per running scenario per mode the name convention is as follows:
    MODELNAME_SCENARIO_RUNTYPE.ini
    1. MODELNAME: resnet or ssd-large
    2. SCENARIO: SingleStream, MultiStream, Server or Offline.
    3. RUNTYPE: accuracy, performance or submission

    for example:
        (a1) resnet_SingleStream_accuracy.ini - ini files for resnet50 single stream accuracy check
        (a2) resnet_SingleStream_perfromance.ini - ini files for resnet50 single stream performance run
        (a3) resnet_SingleStream_submission.ini - ini files for resnet50 single stream submission run, performance run followed by accuracy run.

2. mlpref configuration file configuration file defined by MLPERF requirements in order to achieve a successful and compliant audit
   runs.
    (a) mlperf.config - this file downloaded from mlperf github and must be kept unchanged (contains the base configuration 
        which are mandatory for all scenarios and all run types accuracy, performance or submission)
    (b) user configuration - 
        (b1) user.conf - general user configuration file
        (b2) user_accuracy.conf - user configuration used for accuracy runs.
        (b3) user_perfromance.conf - user configuration used for performance runs
    (c) audit.conf - this file contains audit changes to the configuration for example - enable sampling mode in performance runs.


IMPORTANT: INIT sequence 
The init sequence is defined by the following files - INI file, mlperf.config, user.conf, audit.conf
the INI file contains the specific names of mlperf.config, user.conf and audit.conf so running specific scenario the user must specify 
just the ini file (which specifies all other configuration files).
there exists an override sequence between the different configuration files also user must note that some files are mandatory:
        1. base configuration - defined by mlperf.config - this file is mandatory and must contains valid setup as defined in MLPERF github DB
        2. user configuration - this file contains user specific configuration that is needed to support the user implementation of 
           different running scenarios (SingleStream, MultiStream, Server and Offline), the user configuration file is mandatory 
           including valid setup.
        3. audit.conf - this configuration file is needed for audit run (self-audit scripts to verify compliance with MLPERF   
           requirements), this file is not mandatory for the benchmark run, but user how wishes to avoid error in the 
           mlperf_log_detail.txt) can specify and empty audit.conf file.



To run a single scenario on the benchmark application, use the following command line

./HABANA_benchmark_app ssd-large_Offline_submission.ini 


The above will run the benchmark using ssd-resnet34 model om offline mode.

please note: to control the running mode -AccuracyOnly, PerformanceOnly or SubmissionRun - check the running mode in the user.conf file.



*****   running submission or audit runs is possible by utilizing ******

User may run the full submission and audit runs by running the following bash script  run_full_submission_and_audits.sh, this will run all scenarios for all models (resnet50 and resnet34-large) and generate a submission directory following by an audit run on the submission directory.

Important:
the full submission and audit run take about 13 hours of runs
