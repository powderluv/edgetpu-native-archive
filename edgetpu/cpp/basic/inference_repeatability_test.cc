#include <cmath>

#include "edgetpu/cpp/test_utils.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

DEFINE_int32(stress_test_runs, 500, "Number of iterations for stress test.");

namespace coral {
namespace {

TEST(InferenceRepeatabilityTest, MobilenetV1) {
  RepeatabilityTest(ModelPath("mobilenet_v1_1.0_224_quant_edgetpu.tflite"),
                    FLAGS_stress_test_runs);
}

TEST(InferenceRepeatabilityTest, MobilenetV1SSD) {
  RepeatabilityTest(
      ModelPath("mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite"),
      FLAGS_stress_test_runs);
}

TEST(InferenceRepeatabilityTest, InceptionV2) {
  RepeatabilityTest(ModelPath("inception_v2_224_quant_edgetpu.tflite"),
                    FLAGS_stress_test_runs);
}

TEST(InferenceRepeatabilityTest, InceptionV4) {
  RepeatabilityTest(ModelPath("inception_v4_299_quant_edgetpu.tflite"),
                    FLAGS_stress_test_runs);
}

}  // namespace
}  // namespace coral

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
