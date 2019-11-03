from setuptools import Extension
from setuptools import setup
import glob

#public_headers = [
#    "./src/schedule/schedule.h",
#    "./src/schedule/settings/mlperf_settings.h",
#]

lib_headers = glob.glob(r"./src/**/*.h", recursive=True) + \
    glob.glob(r"./src/**/*.hpp", recursive=True)

lib_sources = glob.glob(r"./src/**/*.cpp", recursive=True) + \
    glob.glob(r"./src/**/*.cc", recursive=True)

execlude_headers = glob.glob(r"./src/*.h", recursive=False) + \
    glob.glob(r"./src/*.hpp", recursive=False) + \
    glob.glob(r"./src/third_party/**/*.h", recursive=True) + \
    glob.glob(r"./src/third_party/**/*.hpp", recursive=True) + \
    glob.glob(r"./src/loadgen/tests/*.h", recursive=False) + \
    glob.glob(r"./src/loadgen/tests/*.hpp", recursive=False)
#    glob.glob(r"./src/loadgen/**/*.h", recursive=True) + \
#    glob.glob(r"./src/loadgen/**/*.hpp", recursive=True)

execlude_sources = glob.glob(r"./src/*.cpp", recursive=False) + \
    glob.glob(r"./src/*.cc", recursive=False) + \
    glob.glob(r"./src/third_party/**/*.cpp", recursive=True) + \
    glob.glob(r"./src/third_party/**/*.cc", recursive=True) + \
    glob.glob(r"./src/loadgen/tests/*.cpp", recursive=False) + \
    glob.glob(r"./src/loadgen/tests/*.cc", recursive=False) + \
    glob.glob(r"./src/cnpy/example1.cpp", recursive=False)
#    glob.glob(r"./src/loadgen/**/*.cpp", recursive=True) + \
#    glob.glob(r"./src/loadgen/**/*.cc", recursive=True)

#lib_bindings = [
#    "./src/bindings/python_api.cpp",
#]

mlperf_schedule_headers = list(set(lib_headers) - set(execlude_headers))
mlperf_schedule_sources = list(set(lib_sources) - set(execlude_sources))

mlperf_schedule_headers.sort()
mlperf_schedule_sources.sort()

mlperf_schedule_module = Extension(
        "mlperf_schedule",
#        define_macros=[("MAJOR_VERSION", "1"),
#                       ("MINOR_VERSION", "0"),
#                      ],
        include_dirs=[".",
#                     "./src/third_party/loadgen",
#                     "./src/third_party/loadgen/bindings",
                      "/usr/include",
#                     "/usr/local/include/boost",
                      "/usr/local/include/opencv4",
                      "/usr/local/cuda/include",
                      "./src/third_party/pybind/include",
                      "/usr/include/x86_64-linux-gnu/include",
                      "/root/wsh/TensorRT/TensorRT-6.0.1.5/samples/samplePlugin",
                     ],
        library_dirs=[
#           "./src/third_party/loadgen/lib",
            "/lib/x86_64-linux-gnu",
            "/usr/local/lib",
            "/usr/local/cuda/lib64",
            "/usr/lib/x86_64-linux-gnu",
           ],
        sources=mlperf_schedule_sources,
        depends=mlperf_schedule_headers,
        libraries=[
#                  "boost_thread",
                   "opencv_core",
                   "opencv_highgui",
                   "opencv_imgproc",
                   "cudart",
                   "cudnn",
                   "cublas",
                   "nvinfer",
                   "nvinfer_plugin",
                   "nvparsers",
                   "z",
#                   "mlperf_loadgen",
               ])

setup(name="mlperf_schedule",
      version="1.0",
      description="MLPerf Inference Schedule python bindings",
      url="",
      ext_modules=[mlperf_schedule_module])

