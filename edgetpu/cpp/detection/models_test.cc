#include "edgetpu/cpp/test_utils.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace coral {
namespace {

TEST(DetectionEngineTest, TestSSDModelsWithCat) {
  // Mobilenet V1 SSD.
  // 4 tensors are returned after post processing operator.
  //
  // 1: detected bounding boxes;
  // 2: detected class label;
  // 3: detected score;
  // 4: number of detected objects;
  TestCatMsCocoDetection(
      ModelPath("mobilenet_ssd_v1_coco_quant_postprocess.tflite"),
      /*score_threshold=*/0.79, /*iou_threshold=*/0.8);
  TestCatMsCocoDetection(
      ModelPath("mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite"),
      /*score_threshold=*/0.79, /*iou_threshold=*/0.8);

  // Mobilenet V2 SSD
  TestCatMsCocoDetection(
      ModelPath("mobilenet_ssd_v2_coco_quant_postprocess.tflite"),
      /*score_threshold=*/0.96, /*iou_threshold=*/0.86);
  TestCatMsCocoDetection(
      ModelPath("mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"),
      /*score_threshold=*/0.96, /*iou_threshold=*/0.86);
}

void TestFaceDetection(const std::string& model_name, float score_threshold,
                       float iou_threshold) {
  TestDetection(ModelPath(model_name), TestDataPath("grace_hopper.bmp"),
                /*expected_box=*/{0.29, 0.21, 0.74, 0.57}, /*expected_label=*/0,
                score_threshold, iou_threshold);
}

TEST(DetectionEngineTest, TestFaceModel) {
  TestFaceDetection("mobilenet_ssd_v2_face_quant_postprocess.tflite", 0.7,
                    0.65);
  TestFaceDetection("mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite",
                    0.7, 0.65);
}

}  // namespace
}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
