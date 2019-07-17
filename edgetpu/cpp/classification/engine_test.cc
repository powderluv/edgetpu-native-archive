#include "edgetpu/cpp/classification/engine.h"

#include "edgetpu/cpp/test_utils.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace coral {
namespace {

using ::testing::ElementsAre;

TEST(ClassificationEngineTest, ClassificationCandidate) {
  ClassificationCandidate a(1, 0.2), b(1, 0.5);
  // Equal.
  EXPECT_TRUE(a == ClassificationCandidate(1, 0.2));
  EXPECT_FALSE(a == ClassificationCandidate(1, 0.19));
  EXPECT_FALSE(a == ClassificationCandidate(2, 0.2));
  // Not Equal.
  EXPECT_TRUE(a != b);
  EXPECT_FALSE(a == b);
  // Assign.
  ClassificationCandidate tmp(5, 0.7);
  EXPECT_TRUE(a != tmp);
  tmp = a;
  EXPECT_TRUE(a == tmp);
  tmp = b;
  EXPECT_TRUE(a != tmp);
}

TEST(ClassificationEngineTest, TestDebugFunctions) {
  // Load the model.
  ClassificationEngine mobilenet_engine(
      ModelPath("mobilenet_v1_1.0_224_quant.tflite"));
  // Check input dimensions.
  std::vector<int> input_tensor_shape =
      mobilenet_engine.get_input_tensor_shape();
  EXPECT_EQ(4, input_tensor_shape.size());
  EXPECT_THAT(input_tensor_shape, ElementsAre(1, 224, 224, 3));
  // Check output tensors.
  std::vector<int> output_tensor_sizes =
      mobilenet_engine.get_all_output_tensors_sizes();
  EXPECT_EQ(1, output_tensor_sizes.size());
  EXPECT_EQ(1001, output_tensor_sizes[0]);
  // Check model's path.
  EXPECT_EQ(ModelPath("mobilenet_v1_1.0_224_quant.tflite"),
            mobilenet_engine.model_path());
}

}  // namespace
}  // namespace coral

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
