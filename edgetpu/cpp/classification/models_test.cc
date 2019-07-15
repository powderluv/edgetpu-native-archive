#include "edgetpu/cpp/classification/engine.h"
#include "edgetpu/cpp/test_utils.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace coral {
namespace {

TEST(ClassificationEngineTest, TestMobilenetModels) {
  // Mobilenet V1 1.0
  TestClassification(ModelPath("mobilenet_v1_1.0_224_quant.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.78,
                     /*expected_top1_label=*/286);  // Egyptian cat
  TestClassification(ModelPath("mobilenet_v1_1.0_224_quant_edgetpu.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.78,
                     /*expected_top1_label=*/286);  // Egyptian cat

  // Mobilenet V1 0.25
  TestClassification(ModelPath("mobilenet_v1_0.25_128_quant.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.36,
                     /*expected_top1_label=*/283);  // tiger cat
  TestClassification(ModelPath("mobilenet_v1_0.25_128_quant_edgetpu.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.36,
                     /*expected_top1_label=*/283);  // tiger cat

  // Mobilenet V1 0.5
  TestClassification(ModelPath("mobilenet_v1_0.5_160_quant.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.68,
                     /*expected_top1_label=*/286);  // Egyptian cat
  TestClassification(ModelPath("mobilenet_v1_0.5_160_quant_edgetpu.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.68,
                     /*expected_top1_label=*/286);  // Egyptian cat
  // Mobilenet V1 0.75
  TestClassification(ModelPath("mobilenet_v1_0.75_192_quant.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.4,
                     /*expected_top1_label=*/283);  // tiger cat
  TestClassification(ModelPath("mobilenet_v1_0.75_192_quant_edgetpu.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.4,
                     /*expected_top1_label=*/283);  // tiger cat

  // Mobilenet V2
  TestClassification(ModelPath("mobilenet_v2_1.0_224_quant.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.81,
                     /*expected_top1_label=*/286);  // Egyptian cat
  TestClassification(ModelPath("mobilenet_v2_1.0_224_quant_edgetpu.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.79,
                     /*expected_top1_label=*/286);  // Egyptian cat
}

TEST(ClassificationEngineTest, TestInceptionModels) {
  // Inception V1
  TestClassification(ModelPath("inception_v1_224_quant.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.37,
                     /*expected_top1_label=*/282);  // tabby, tabby cat

  TestClassification(ModelPath("inception_v1_224_quant_edgetpu.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.38,
                     /*expected_top1_label=*/286);  // Egyptian cat

  // Inception V2
  TestClassification(ModelPath("inception_v2_224_quant.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.65,
                     /*expected_top1_label=*/286);  // Egyptian cat
  TestClassification(ModelPath("inception_v2_224_quant_edgetpu.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.61,
                     /*expected_top1_label=*/286);  // Egyptian cat

  // Inception V3
  TestClassification(ModelPath("inception_v3_299_quant.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.58,
                     /*expected_top1_label=*/282);  // tabby, tabby cat
  TestClassification(ModelPath("inception_v3_299_quant_edgetpu.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.597,
                     /*expected_top1_label=*/282);  // tabby, tabby cat

  // Inception V4
  TestClassification(ModelPath("inception_v4_299_quant.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.35,
                     /*expected_top1_label=*/286);  // Egyptian cat
  TestClassification(ModelPath("inception_v4_299_quant_edgetpu.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.41,
                     /*expected_top1_label=*/282);  // tabby, tabby cat
}

TEST(ClassificationEngineTest, TestINatModels) {
  // Plant model
  TestClassification(
      ModelPath("mobilenet_v2_1.0_224_inat_plant_quant.tflite"),
      TestDataPath("sunflower.bmp"),
      /*score_threshold=*/0.4,
      /*expected_top1_label=*/1680);  // Helianthus annuus (common sunflower)
  TestClassification(
      ModelPath("mobilenet_v2_1.0_224_inat_plant_quant_edgetpu.tflite"),
      TestDataPath("sunflower.bmp"),
      /*score_threshold=*/0.4,
      /*expected_top1_label=*/1680);  // Helianthus annuus (common sunflower)

  // Insect model
  TestClassification(ModelPath("mobilenet_v2_1.0_224_inat_insect_quant.tflite"),
                     TestDataPath("dragonfly.bmp"), /*score_threshold=*/0.2,
                     /*expected_top1_label=*/912);  // Thornbush Dasher
  TestClassification(
      ModelPath("mobilenet_v2_1.0_224_inat_insect_quant_edgetpu.tflite"),
      TestDataPath("dragonfly.bmp"), /*score_threshold=*/0.2,
      /*expected_top1_label=*/912);  // Thornbush Dasher

  // Bird model
  TestClassification(ModelPath("mobilenet_v2_1.0_224_inat_bird_quant.tflite"),
                     TestDataPath("bird.bmp"),
                     /*score_threshold=*/0.8,
                     /*expected_top1_label=*/91);  // White-throated Sparrow
  TestClassification(
      ModelPath("mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite"),
      TestDataPath("bird.bmp"), /*score_threshold=*/0.8,
      /*expected_top1_label=*/91);  // White-throated Sparrow
}

TEST(ClassificationEngineTest, TestEdgeTpuNetModelsCustomPreprocessing) {
  const int kTopk = 3;
  // Custom preprocessing is done by:
  // (v - (mean - zero_point * scale * stddev)) / (stddev * scale)
  {
    // mean 127, stddev 128
    // first input tensor scale: 0.011584, zero_point: 125
    const float effective_scale = 128 * 0.011584;
    const std::vector<float> effective_means(3, 127 - 125 * effective_scale);
    TestClassification(ModelPath("edgetpu_net_small_quant.tflite"),
                       TestDataPath("cat.bmp"), effective_scale,
                       effective_means,
                       /*score_threshold=*/0.4, kTopk,
                       /*expected_topk_label=*/286);  // Egyptian cat
    TestClassification(ModelPath("edgetpu_net_small_quant_edgetpu.tflite"),
                       TestDataPath("cat.bmp"), effective_scale,
                       effective_means,
                       /*score_threshold=*/0.4, kTopk,
                       /*expected_topk_label=*/286);  // Egyptian cat
  }

  {
    // mean 127, stddev 128
    // first input tensor scale: 0.012087, zero_point: 131
    const float effective_scale = 128 * 0.012087;
    const std::vector<float> effective_means(3, 127 - 131 * effective_scale);
    TestClassification(ModelPath("edgetpu_net_medium_quant.tflite"),
                       TestDataPath("cat.bmp"), effective_scale,
                       effective_means,
                       /*score_threshold=*/0.6, kTopk,
                       /*expected_topk_label=*/286);  // Egyptian cat
    TestClassification(ModelPath("edgetpu_net_medium_quant_edgetpu.tflite"),
                       TestDataPath("cat.bmp"), effective_scale,
                       effective_means,
                       /*score_threshold=*/0.6, kTopk,
                       /*expected_topk_label=*/286);  // Egyptian cat
  }

  {
    // mean 127, stddev 128
    // first input tensor scale: 0.012279, zero_point: 130
    const float effective_scale = 128 * 0.012279;
    const std::vector<float> effective_means(3, 127 - 130 * effective_scale);
    TestClassification(ModelPath("edgetpu_net_large_quant.tflite"),
                       TestDataPath("cat.bmp"), effective_scale,
                       effective_means,
                       /*score_threshold=*/0.6, kTopk,
                       /*expected_topk_label=*/286);  // Egyptian cat
    TestClassification(ModelPath("edgetpu_net_large_quant_edgetpu.tflite"),
                       TestDataPath("cat.bmp"), effective_scale,
                       effective_means,
                       /*score_threshold=*/0.6, kTopk,
                       /*expected_topk_label=*/286);  // Egyptian cat
  }
}
}  // namespace
}  // namespace coral

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
