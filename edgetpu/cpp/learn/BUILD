package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = [
        "utils.h",
    ],
    deps = [
        "//edgetpu/cpp:error_reporter",
        "//edgetpu/cpp:utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_glog//:glog",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "utils_test",
    srcs = [
        "utils_test.cc",
    ],
    data = [
        "//edgetpu/cpp/basic/test_data:models",
        "//edgetpu/cpp/learn/imprinting/test_data:models",
    ],
    deps = [
        ":utils",
        "//edgetpu/cpp:error_reporter",
        "//edgetpu/cpp:test_utils",
        "//edgetpu/cpp/basic:basic_engine",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)
