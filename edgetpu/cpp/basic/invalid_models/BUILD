package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "fake_op",
    srcs = [
        "fake_op.cc",
    ],
    hdrs = [
        "fake_op.h",
    ],
    deps = [
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
    ],
)

filegroup(
    name = "models",
    srcs = glob(["*.tflite"]),
)
