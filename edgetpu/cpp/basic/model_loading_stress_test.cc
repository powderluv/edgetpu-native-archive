#include <cmath>

#include "edgetpu/cpp/basic/basic_engine.h"
#include "edgetpu/cpp/test_utils.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

DEFINE_int32(stress_test_runs, 500, "Number of iterations for stress test.");

namespace coral {
namespace {

TEST(ModelLoadingStressTest, AlternateEdgeTpuModels) {
  const std::vector<std::string> model_names = {
      "mobilenet_v1_1.0_224_quant_edgetpu.tflite",
      "mobilenet_v2_1.0_224_quant_edgetpu.tflite",
      "mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite",
      "mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite",
      "inception_v1_224_quant_edgetpu.tflite",
      "inception_v2_224_quant_edgetpu.tflite",
      "inception_v3_299_quant_edgetpu.tflite",
      "inception_v4_299_quant_edgetpu.tflite",
  };

  for (int i = 0; i < FLAGS_stress_test_runs; ++i) {
    VLOG_EVERY_N(0, 100) << "Stress test iter " << i << "...";
    for (int j = 0; j < model_names.size(); ++j) {
      BasicEngine engine(ModelPath(model_names[j]));
    }
  }
}

TEST(ModelLoadingStressTest, AlternateCpuModels) {
  const std::vector<std::string> model_names = {
      "mobilenet_v1_1.0_224_quant.tflite",
      "mobilenet_v2_1.0_224_quant.tflite",
      "mobilenet_ssd_v1_coco_quant_postprocess.tflite",
      "mobilenet_ssd_v2_coco_quant_postprocess.tflite",
      "inception_v1_224_quant.tflite",
      "inception_v2_224_quant.tflite",
      "inception_v3_299_quant.tflite",
      "inception_v4_299_quant.tflite",
  };

  for (int i = 0; i < FLAGS_stress_test_runs; ++i) {
    VLOG_EVERY_N(0, 100) << "Stress test iter " << i << "...";
    for (int j = 0; j < model_names.size(); ++j) {
      BasicEngine engine(ModelPath(model_names[j]));
    }
  }
}

}  // namespace
}  // namespace coral

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
