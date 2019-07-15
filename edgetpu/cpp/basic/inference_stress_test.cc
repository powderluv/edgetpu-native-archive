#include <cmath>

#include "edgetpu/cpp/test_utils.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

DEFINE_int32(stress_test_runs, 500, "Number of iterations for stress test.");
DEFINE_int32(stress_with_sleep_test_runs, 200,
             "Number of iterations for stress test.");
DEFINE_int32(stress_sleep_sec, 3,
             "Seconds to sleep in-between inference runs.");

namespace coral {
namespace {

TEST(InferenceStressTest, MobilenetV1) {
  InferenceStressTest(ModelPath("mobilenet_v1_1.0_224_quant_edgetpu.tflite"),
                      FLAGS_stress_test_runs);
}

TEST(InferenceStressTest, MobilenetV1SSD) {
  InferenceStressTest(
      ModelPath("mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite"),
      FLAGS_stress_test_runs);
}

TEST(InferenceStressTest, InceptionV2) {
  InferenceStressTest(ModelPath("inception_v2_224_quant_edgetpu.tflite"),
                      FLAGS_stress_test_runs);
}

TEST(InferenceStressTest, InceptionV4) {
  InferenceStressTest(ModelPath("inception_v4_299_quant_edgetpu.tflite"),
                      FLAGS_stress_test_runs);
}

// Stress tests with sleep in-between inference runs.
// We cap the runs here as they will take a lot of time to finish.
TEST(InferenceStressTest, MobilenetV1_WithSleep) {
  InferenceStressTest(ModelPath("mobilenet_v1_1.0_224_quant_edgetpu.tflite"),
                      FLAGS_stress_with_sleep_test_runs,
                      FLAGS_stress_sleep_sec);
}

TEST(InferenceStressTest, InceptionV2_WithSleep) {
  InferenceStressTest(ModelPath("inception_v2_224_quant_edgetpu.tflite"),
                      FLAGS_stress_with_sleep_test_runs,
                      FLAGS_stress_sleep_sec);
}

}  // namespace
}  // namespace coral

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
