# NVIDIA MLPerf Inference Benchmarks

## List of Benchmarks

Please refer to the `README.md` in each benchmark directory for implementation details.
- [resnet](resnet/tensorrt)
- [mobilenet](mobilenet/tensorrt)
- [ssd-large](ssd-large/tensorrt)
- [ssd-small](ssd-small/tensorrt)
- [gnmt](gnmt/tensorrt)

## Other Directories

- [common](common) - holds scripts to generate TensorRT optimized plan files and to run the harnesses.
- [harness](harness) - holds source codes of the harness interfacing with LoadGen.
- [plugin](plugin) - holds source codes of TensorRT plugins shared by multiple benchmarks.
