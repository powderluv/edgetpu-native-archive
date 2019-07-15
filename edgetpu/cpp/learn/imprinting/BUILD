package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "imprinting_test_base",
    testonly = 1,
    srcs = [
        "imprinting_test_base.cc",
    ],
    hdrs = [
        "imprinting_test_base.h",
    ],
    deps = [
        "//edgetpu/cpp/basic:basic_engine",
        "//edgetpu/cpp/basic:inference_utils",
        "//edgetpu/cpp/learn:utils",
        "@com_google_absl//absl/strings",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "engine",
    srcs = [
        "engine.cc",
    ],
    hdrs = [
        "engine.h",
    ],
    deps = [
        ":engine_native",
        "@com_google_glog//:glog",
    ],
)

cc_test(
    name = "engine_test",
    timeout = "long",
    srcs = [
        "engine_test.cc",
    ],
    data = [
        "//edgetpu/cpp/learn/imprinting/test_data:images",
        "//edgetpu/cpp/learn/imprinting/test_data:models",
    ],
    deps = [
        ":engine",
        ":imprinting_test_base",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "engine_native",
    srcs = [
        "engine_native.cc",
    ],
    hdrs = [
        "engine_native.h",
    ],
    deps = [
        "//edgetpu/cpp:error_reporter",
        "//edgetpu/cpp/basic:basic_engine_native",
        "//edgetpu/cpp/learn:utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_glog//:glog",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "engine_native_test",
    timeout = "long",
    srcs = [
        "engine_native_test.cc",
    ],
    data = [
        "//edgetpu/cpp/learn/imprinting/test_data:images",
        "//edgetpu/cpp/learn/imprinting/test_data:models",
    ],
    deps = [
        ":engine_native",
        ":imprinting_test_base",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest",
    ],
)
