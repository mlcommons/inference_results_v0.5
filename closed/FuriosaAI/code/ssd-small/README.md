# Running `ssd-mobilenet` on Furiosa AI's Renegade accelerator

## Requirements
- Pyhton3+
- absl-py
- flatbuffers (flatc)
- cmake
- bazel
- single-file public domain (or MIT licensed) libraries for C/C++ (https://github.com/nothings/stb)
- internal libraries (libdg.so, libnux.so)

## Instructions

1. git clone https://github.com/mlperf/inference
2. copy libdg.so, libnux.so, libtensorflowlite.so to ./build/bin
3. copy dg.h to ./build/include
4. run ./build.sh
5. resize images using [resize-coco.py](furiosa-loadgen/resize-coco.py) & make those as bin file using [coco-preprocess.py](furiosa-loadgen/coco-preprocess.py)
6. run backgraound daemons, `xdma.ko`, `npud`
7. run built loadgen binaries with dynamically linked libraries (libdg.so, libnux.so) and npud (NPU daemon)
