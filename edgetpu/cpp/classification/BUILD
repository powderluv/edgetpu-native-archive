package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "engine",
    srcs = [
        "engine.cc",
    ],
    hdrs = [
        "engine.h",
    ],
    deps = [
        "//edgetpu/cpp/basic:basic_engine",
        "@com_google_glog//:glog",
    ],
)

cc_test(
    name = "engine_test",
    srcs = [
        "engine_test.cc",
    ],
    data = [
        "//edgetpu/cpp/basic/test_data:models",
    ],
    deps = [
        ":engine",
        "//edgetpu/cpp:test_utils",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "models_test",
    srcs = [
        "models_test.cc",
    ],
    data = [
        "//edgetpu/cpp/basic/test_data:images",
        "//edgetpu/cpp/basic/test_data:models",
    ],
    deps = [
        ":engine",
        "//edgetpu/cpp:test_utils",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest",
    ],
)
