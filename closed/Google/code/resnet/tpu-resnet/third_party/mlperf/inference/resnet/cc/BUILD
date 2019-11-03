package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache License 2.0

exports_files(["LICENSE"])

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    linkopts = [
        "-lm",
        "-lpthread",
        "-lrt",
        "-Wl,--allow-multiple-definition",
        "-Wl,--whole-archive",
    ],
    deps = [
        "//mlperf/inference/loadgen",
        "//mlperf/inference/runner",
        "//mlperf/inference/runner:options",
        "//mlperf/inference/runner:query_sample_library_tpu",
        "//mlperf/inference/runner:system_under_test_tpu",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/core",
    ],
)
