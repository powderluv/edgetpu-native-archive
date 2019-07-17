#include "edgetpu/cpp/detection/engine.h"

#include "edgetpu/cpp/test_utils.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace coral {
namespace {

using ::testing::ElementsAre;

TEST(DetectionEngineTest, TestDetectionCandidate) {
  DetectionCandidate a({1, 0.2, {0.0, 0.0, 1.0, 1.0}}),
      b({1, 0.5, {0.0, 0.0, 1.0, 1.0}});
  // Equal.
  EXPECT_TRUE(a == DetectionCandidate({1, 0.2, {0.0, 0.0, 1.0, 1.0}}));
  EXPECT_FALSE(a == DetectionCandidate({1, 0.2, {0.1, 0.1, 1.0, 1.0}}));
  EXPECT_FALSE(a == DetectionCandidate({1, 0.2, {0.0, 0.0, 0.9, 0.9}}));
  EXPECT_FALSE(a == DetectionCandidate({1, 0.19, {0.0, 0.0, 1.0, 1.0}}));
  EXPECT_FALSE(a == DetectionCandidate({2, 0.2, {0.0, 0.0, 1.0, 1.0}}));
  EXPECT_TRUE(a != b);
  EXPECT_FALSE(a == b);
  // Assign.
  DetectionCandidate tmp({5, 0.7, {0.0, 0.0, 0.0, 0.0}});
  EXPECT_TRUE(a != tmp);
  tmp = a;
  EXPECT_TRUE(a == tmp);
  tmp = b;
  EXPECT_TRUE(a != tmp);
}

TEST(DetectionEngineTest, TestDebugFunctions) {
  // Load the model.
  DetectionEngine engine(
      ModelPath("mobilenet_ssd_v1_coco_quant_postprocess.tflite"));
  // Check input dimensions.
  std::vector<int> input_tensor_shape = engine.get_input_tensor_shape();
  EXPECT_EQ(4, input_tensor_shape.size());
  EXPECT_THAT(input_tensor_shape, ElementsAre(1, 300, 300, 3));
  // Check output tensors.
  std::vector<int> output_tensor_sizes = engine.get_all_output_tensors_sizes();
  EXPECT_EQ(4, output_tensor_sizes.size());
  EXPECT_THAT(output_tensor_sizes, ElementsAre(80, 20, 20, 1));
  // Check model's path.
  EXPECT_EQ(ModelPath("mobilenet_ssd_v1_coco_quant_postprocess.tflite"),
            engine.model_path());
}

}  // namespace
}  // namespace coral

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
