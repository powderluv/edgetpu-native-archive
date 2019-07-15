package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "basic_engine_native",
    srcs = [
        "basic_engine_native.cc",
    ],
    hdrs = [
        "basic_engine_native.h",
    ],
    deps = [
        ":edgetpu_resource_manager",
        ":inference_utils",
        "//edgetpu/cpp:error_reporter",
        "@com_google_absl//absl/memory",
        "@libedgetpu//:header",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "basic_engine",
    srcs = [
        "basic_engine.cc",
    ],
    hdrs = [
        "basic_engine.h",
    ],
    deps = [
        ":basic_engine_native",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "inference_utils",
    srcs = [
        "inference_utils.cc",
    ],
    hdrs = [
        "inference_utils.h",
    ],
    deps = [
        "//edgetpu/cpp:error_reporter",
        "//edgetpu/cpp:utils",
        "//edgetpu/cpp/posenet:posenet_decoder_op",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_glog//:glog",
        "@libedgetpu//:header",
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_test(
    name = "basic_engine_native_test",
    timeout = "long",
    srcs = [
        "basic_engine_native_test.cc",
    ],
    data = [
        "//edgetpu/cpp/basic/invalid_models:models",
        "//edgetpu/cpp/basic/test_data:images",
        "//edgetpu/cpp/basic/test_data:models",
    ],
    deps = [
        ":basic_engine_native",
        "//edgetpu/cpp:test_utils",
        "//edgetpu/cpp/basic/invalid_models:fake_op",
        "@com_google_absl//absl/memory",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "basic_engine_test",
    timeout = "long",
    srcs = [
        "basic_engine_test.cc",
    ],
    data = [
        "//edgetpu/cpp/basic/invalid_models:models",
        "//edgetpu/cpp/basic/test_data:images",
        "//edgetpu/cpp/basic/test_data:models",
    ],
    deps = [
        ":basic_engine",
        "//edgetpu/cpp:test_utils",
        "//edgetpu/cpp/basic/invalid_models:fake_op",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "models_test",
    timeout = "long",
    srcs = [
        "models_test.cc",
    ],
    data = [
        "//edgetpu/cpp/basic/test_data:images",
        "//edgetpu/cpp/basic/test_data:models",
    ],
    deps = [
        ":basic_engine",
        ":inference_utils",
        "//edgetpu/cpp:test_utils",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "edgetpu_resource_manager",
    srcs = [
        "edgetpu_resource_manager.cc",
    ],
    hdrs = [
        "edgetpu_resource_manager.h",
    ],
    deps = [
        "//edgetpu/cpp:error_reporter",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_glog//:glog",
        "@libedgetpu//:header",
        "@libedgetpu//:lib",
    ],
)

cc_test(
    name = "edgetpu_resource_manager_test",
    srcs = [
        "edgetpu_resource_manager_test.cc",
    ],
    deps = [
        ":edgetpu_resource_manager",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest",
    ],
)

cc_binary(
    name = "models_benchmark",
    testonly = 1,
    srcs = [
        "models_benchmark.cc",
    ],
    data = [
        "//edgetpu/cpp/basic/test_data:images",
        "//edgetpu/cpp/basic/test_data:models",
    ],
    deps = [
        ":basic_engine",
        "//edgetpu/cpp:test_utils",
        "@com_github_google_benchmark//:benchmark",
        "@com_github_google_benchmark//:benchmark_main",
        "@com_google_glog//:glog",
    ],
)

cc_test(
    name = "inference_repeatability_test",
    timeout = "long",
    srcs = [
        "inference_repeatability_test.cc",
    ],
    data = [
        "//edgetpu/cpp/basic/test_data:models",
    ],
    deps = [
        ":basic_engine",
        "//edgetpu/cpp:test_utils",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "model_loading_stress_test",
    timeout = "long",
    srcs = [
        "model_loading_stress_test.cc",
    ],
    data = [
        "//edgetpu/cpp/basic/test_data:models",
    ],
    deps = [
        ":basic_engine",
        "//edgetpu/cpp:test_utils",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "inference_stress_test",
    timeout = "long",
    srcs = [
        "inference_stress_test.cc",
    ],
    data = [
        "//edgetpu/cpp/basic/test_data:models",
    ],
    deps = [
        "//edgetpu/cpp:test_utils",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest",
    ],
)

cc_binary(
    name = "edgetpu_resource_manager_benchmark",
    testonly = 1,
    srcs = ["edgetpu_resource_manager_benchmark.cc"],
    copts = [
        "-DBENCHMARK_DISABLE_THREADS",
    ],
    deps = [
        ":edgetpu_resource_manager",
        "@com_github_google_benchmark//:benchmark",
        "@com_github_google_benchmark//:benchmark_main",
        "@com_google_glog//:glog",
        "@libedgetpu//:header",
    ],
)

cc_test(
    name = "multiple_tpus_inference_stress_test",
    srcs = [
        "multiple_tpus_inference_stress_test.cc",
    ],
    data = [
        "//edgetpu/cpp/basic/test_data:images",
        "//edgetpu/cpp/basic/test_data:models",
    ],
    deps = [
        ":inference_utils",
        "//edgetpu/cpp:test_utils",
        "//edgetpu/cpp/basic:basic_engine",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest",
    ],
)

cc_binary(
    name = "multiple_tpus_performance_analysis",
    testonly = 1,
    srcs = ["multiple_tpus_performance_analysis.cc"],
    data = [
        "//edgetpu/cpp/basic/test_data:images",
        "//edgetpu/cpp/basic/test_data:models",
    ],
    deps = [
        ":inference_utils",
        "//edgetpu/cpp:test_utils",
        "//edgetpu/cpp/basic:basic_engine",
        "@com_google_glog//:glog",
    ],
)
