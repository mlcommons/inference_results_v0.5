# TODO(wangtao): describe this package.
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Simplified BSD

exports_files(["LICENSE"])

cc_library(
    name = "logging_h",
    hdrs = [
        "logging.h",
        "query_sample.h",
    ],
    copts = ["-Wno-c++98-c++11-compat"],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = [
        "query_sample.h",
        "utils.h",
    ],
    copts = ["-Wno-c++98-c++11-compat"],
    deps = [
        ":logging_h",
    ],
)

cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    copts = ["-Wno-c++98-c++11-compat"],
    deps = [
        ":logging_h",
        ":utils",
    ],
)

cc_library(
    name = "version",
    srcs = [
        "version.cc",
        "version_generated.cc",
    ],
    hdrs = [
        "version.h",
    ],
    copts = ["-Wno-c++98-c++11-compat"],
    deps = [
        ":logging",
        ":logging_h",
    ],
)

cc_library(
    name = "test_settings_internal",
    srcs = ["test_settings_internal.cc"],
    hdrs = [
        "test_settings.h",
        "test_settings_internal.h",
    ],
    copts = ["-Wno-c++98-c++11-compat"],
    deps = [
        ":logging",
        ":logging_h",
        ":utils",
    ],
)

cc_library(
    name = "loadgen",
    srcs = ["loadgen.cc"],
    hdrs = [
        "loadgen.h",
        "query_sample.h",
        "query_sample_library.h",
        "system_under_test.h",
        "test_settings.h",
    ],
    copts = ["-Wno-c++98-c++11-compat"],
    deps = [
        ":logging",
        ":logging_h",
        ":test_settings_internal",
        ":utils",
        ":version",
    ],
)
