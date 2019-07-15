package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "tflite_graph_util",
    srcs = ["tflite_graph_util.cc"],
    hdrs = ["tflite_graph_util.h"],
    deps = [
        "//edgetpu/cpp:utils",
        "@com_google_glog//:glog",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_binary(
    name = "join_tflite_models",
    srcs = ["join_tflite_models.cc"],
    deps = [
        ":tflite_graph_util",
        "@com_google_glog//:glog",
    ],
)

cc_test(
    name = "tflite_graph_util_test",
    srcs = ["tflite_graph_util_test.cc"],
    data = [
        ":testdata/mobilenet_quant_v1_1.0_224_partial_delegation.tflite",
        ":testdata/mobilenet_quant_v1_224_feature_layers-custom_op.tflite",
        ":testdata/mobilenet_quant_v1_224_head_layers.tflite",
        ":testdata/mobilenet_ssd_v2_coco_8bit_mini_tensor_feature_layers-custom_op.tflite",
        ":testdata/mobilenet_ssd_v2_coco_8bit_mini_tensor_head_layers.tflite",
        ":testdata/mobilenet_ssd_v2_coco_8bit_mini_tensor_partial_delegation.tflite",
    ],
    deps = [
        ":tflite_graph_util",
        "@com_google_absl//absl/strings",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest",
    ],
)
