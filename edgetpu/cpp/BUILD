package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "error_reporter",
    srcs = [
        "error_reporter.cc",
    ],
    hdrs = [
        "error_reporter.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/lite/core/api",
    ],
)

cc_test(
    name = "error_reporter_test",
    srcs = [
        "error_reporter_test.cc",
    ],
    deps = [
        ":error_reporter",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "version",
    srcs = [
        "version.cc",
    ],
    hdrs = [
        "version.h",
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@libedgetpu//:header",
        "@libedgetpu//:lib",
    ],
)

cc_test(
    name = "version_test",
    srcs = [
        "version_test.cc",
    ],
    deps = [
        ":version",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = [
        "utils.h",
    ],
    deps = [
        "//edgetpu/cpp:error_reporter",
        "@com_google_absl//absl/strings",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "test_utils",
    testonly = 1,
    srcs = [
        "test_utils.cc",
    ],
    hdrs = [
        "test_utils.h",
    ],
    deps = [
        "//edgetpu/cpp/basic:basic_engine",
        "//edgetpu/cpp/basic:inference_utils",
        "//edgetpu/cpp/classification:engine",
        "//edgetpu/cpp/detection:engine",
        "@com_github_gflags_gflags//:gflags",
        "@com_github_google_benchmark//:benchmark",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest",
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
    ],
)
