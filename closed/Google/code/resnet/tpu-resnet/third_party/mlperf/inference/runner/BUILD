package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache License 2.0

exports_files(["LICENSE"])

cc_library(
    name = "queue",
    hdrs = ["queue.h"],
)

cc_library(
    name = "runner",
    srcs = ["runner.cc"],
    hdrs = ["runner.h"],
    deps = [
        ":loader",
        ":options",
        ":queue",
        "//mlperf/inference/loadgen",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core/profiler/lib:traceme",
    ],
)

cc_library(
    name = "system_under_test_tpu",
    hdrs = ["system_under_test_tpu.h"],
    deps = [
        ":runner",
        "//mlperf/inference/loadgen",
    ],
)

cc_library(
    name = "query_sample_library_tpu",
    hdrs = ["query_sample_library_tpu.h"],
    deps = [
        ":dataset",
        ":runner",
        "//mlperf/inference/loadgen",
    ],
)

cc_library(
    name = "options",
    hdrs = ["options.h"],
    deps = [
        "//mlperf/inference/loadgen",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "loader",
    srcs = ["loader.cc"],
    hdrs = ["loader.h"],
    deps = [
        ":options",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/cc/saved_model:tag_constants",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:grpc_session",
        "//tensorflow_serving/batching:batching_session",
    ],
)

cc_library(
    name = "dataset",
    srcs = ["dataset.cc"],
    hdrs = [
        "dataset.h",
    ],
    deps = [
        "//mlperf/inference/loadgen",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)
