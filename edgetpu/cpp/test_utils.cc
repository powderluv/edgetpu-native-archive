#include "edgetpu/cpp/test_utils.h"

#include <dirent.h>
#include <sys/types.h>

#include <fstream>
#include <random>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "edgetpu/cpp/basic/basic_engine.h"
#include "edgetpu/cpp/classification/engine.h"
#include "edgetpu/cpp/detection/engine.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "tensorflow/lite/builtin_op_data.h"

DEFINE_string(model_dir, "edgetpu/cpp/basic/test_data", "Model directory");

DEFINE_string(test_data_dir, "edgetpu/cpp/basic/test_data",
              "Test data directory");

namespace coral {

namespace {
template <typename SrcType, typename DstType>
DstType saturate_cast(SrcType val) {
  if (val > static_cast<SrcType>(std::numeric_limits<DstType>::max())) {
    return std::numeric_limits<DstType>::max();
  }
  if (val < static_cast<SrcType>(std::numeric_limits<DstType>::lowest())) {
    return std::numeric_limits<DstType>::lowest();
  }
  return static_cast<DstType>(val);
}
}  // namespace

std::string ModelPath(const std::string& model_name) {
  if (model_name.empty()) return FLAGS_model_dir;
  return absl::StrCat(FLAGS_model_dir, "/", model_name);
}

std::string TestDataPath(const std::string& name) {
  if (name.empty()) return FLAGS_test_data_dir;
  return absl::StrCat(FLAGS_test_data_dir, "/", name);
}

std::vector<uint8_t> GetRandomInput(const int n) {
  unsigned int seed = 1;
  std::vector<uint8_t> result;
  result.resize(n);
  for (int i = 0; i < n; ++i) {
    result[i] = rand_r(&seed) % 256;
  }
  return result;
}

std::vector<uint8_t> GetRandomInput(std::vector<int> shape) {
  int n = 1;
  for (int i = 0; i < shape.size(); ++i) {
    n *= shape[i];
  }
  return GetRandomInput(n);
}

std::vector<std::string> GetAllModels() {
  DIR* dirp = opendir(ModelPath("").c_str());
  struct dirent* dp;
  std::vector<std::string> ret;
  while ((dp = readdir(dirp)) != nullptr) {
    if (EndsWith(dp->d_name, ".tflite")) ret.push_back(dp->d_name);
  }
  closedir(dirp);
  return ret;
}

void TestWithRandomInput(const std::string& model_path) {
  // Load the model.
  BasicEngine engine(model_path);
  engine.RunInference(GetRandomInput(engine.get_input_tensor_shape()));
}

std::vector<std::vector<float>> TestWithImage(const std::string& model_path,
                                              const std::string& image_path) {
  // Load the model.
  LOG(INFO) << "Testing model: " << model_path;
  BasicEngine engine(model_path);
  std::vector<int> input_tensor_shape = engine.get_input_tensor_shape();
  // Read image.
  std::vector<uint8_t> input = GetInputFromImage(
      image_path,
      {input_tensor_shape[1], input_tensor_shape[2], input_tensor_shape[3]});
  CHECK(!input.empty()) << "Input image path: " << image_path;
  // Get result.
  return engine.RunInference(input);
}

bool TopKContains(const std::vector<ClassificationCandidate>& topk, int label) {
  for (const auto& entry : topk) {
    if (entry.id == label) return true;
  }
  LOG(ERROR) << "Top K results do not contain " << label;
  for (const auto& p : topk) {
    LOG(ERROR) << p.id << ", " << p.score;
  }
  return false;
}

// Tests a classification model with customized preprocessing.
void TestClassification(const std::string& model_path,
                        const std::string& image_path, float effective_scale,
                        const std::vector<float>& effective_means,
                        float score_threshold, int k, int expected_topk_label) {
  LOG(INFO) << "Testing model: " << model_path;
  // Load the model.
  ClassificationEngine engine(model_path);
  std::vector<int> input_tensor_shape = engine.get_input_tensor_shape();
  // Read image.
  std::vector<uint8_t> input_tensor = GetInputFromImage(
      image_path,
      {input_tensor_shape[1], input_tensor_shape[2], input_tensor_shape[3]});

  const int num_channels = effective_means.size();
  for (int i = 0; i < input_tensor.size(); i += num_channels) {
    input_tensor[i] = saturate_cast<float, uint8_t>(
        (input_tensor[i] - effective_means[0]) / effective_scale);
    input_tensor[i + 1] = saturate_cast<float, uint8_t>(
        (input_tensor[i + 1] - effective_means[1]) / effective_scale);
    input_tensor[i + 2] = saturate_cast<float, uint8_t>(
        (input_tensor[i + 2] - effective_means[2]) / effective_scale);
  }

  CHECK(!input_tensor.empty()) << "Input image path: " << image_path;
  EXPECT_TRUE(TopKContains(
      engine.ClassifyWithInputTensor(input_tensor, score_threshold, k),
      expected_topk_label));
}

void TestClassification(const std::string& model_path,
                        const std::string& image_path, float score_threshold,
                        int k, int expected_topk_label) {
  LOG(INFO) << "Testing model: " << model_path;
  // Load the model.
  ClassificationEngine engine(model_path);
  std::vector<int> input_tensor_shape = engine.get_input_tensor_shape();
  // Read image.
  std::vector<uint8_t> input_tensor = GetInputFromImage(
      image_path,
      {input_tensor_shape[1], input_tensor_shape[2], input_tensor_shape[3]});
  CHECK(!input_tensor.empty()) << "Input image path: " << image_path;
  EXPECT_TRUE(TopKContains(
      engine.ClassifyWithInputTensor(input_tensor, score_threshold, k),
      expected_topk_label));
}

void TestClassification(const std::string& model_path,
                        const std::string& image_path, float score_threshold,
                        int expected_top1_label) {
  TestClassification(model_path, image_path, score_threshold, /*k=*/1,
                     expected_top1_label);
}

void TestDetection(const std::string& model_path, const std::string& image_path,
                   const Box& expected_box, int expected_label,
                   float score_threshold, float iou_threshold) {
  DetectionEngine engine(model_path);
  std::vector<int> input_tensor_shape = engine.get_input_tensor_shape();
  // Read image.
  std::vector<uint8_t> input_tensor = GetInputFromImage(
      image_path,
      {input_tensor_shape[1], input_tensor_shape[2], input_tensor_shape[3]});

  auto candiates =
      engine.DetectWithInputTensor(input_tensor, score_threshold, /*top_k=*/1);
  EXPECT_EQ(candiates.size(), 1);
  DetectionCandidate result = candiates[0];
  EXPECT_EQ(result.id, expected_label);
  EXPECT_GT(result.score, score_threshold);
  EXPECT_GT(IntersectionOverUnion(result.bounding_box, expected_box),
            iou_threshold);
}

void TestCatMsCocoDetection(const std::string& model_path,
                            float score_threshold, float iou_threshold) {
  TestDetection(model_path, TestDataPath("cat.bmp"),
                /*expected_box=*/{0.1, 0.1, 0.7, 1.0},
                /*expected_label=*/16, score_threshold, iou_threshold);
}

void BenchmarkModelsOnEdgeTpu(const std::vector<std::string>& model_paths,
                              benchmark::State& state) {
  const int number_models = model_paths.size();
  std::vector<std::unique_ptr<coral::BasicEngine>> engines;
  std::vector<std::vector<uint8_t>> inputs;
  for (int model_index = 0; model_index < number_models; ++model_index) {
    const auto& model_path = model_paths[model_index];
    std::unique_ptr<coral::BasicEngine> engine;
    if (model_index == 0) {
      engine = absl::make_unique<coral::BasicEngine>(model_path);
    } else {
      // Engines should run on the same EdgeTpu device.
      engine = absl::make_unique<coral::BasicEngine>(model_path,
                                                     engines[0]->device_path());
    }
    const auto& model_input = GetRandomInput(engine->get_input_tensor_shape());
    inputs.push_back(model_input);
    engines.push_back(std::move(engine));
  }
  while (state.KeepRunning()) {
    for (int i = 0; i < engines.size(); ++i) {
      engines[i]->RunInference(inputs[i]);
    }
  }
}

void BenchmarkModelOnEdgeTpu(const std::string& model_path,
                             benchmark::State& state) {
  BenchmarkModelsOnEdgeTpu({model_path}, state);
}

void RepeatabilityTest(const std::string& model_path, int runs) {
  BasicEngine engine(model_path);
  const auto& input_data = GetRandomInput(engine.get_input_tensor_shape());
  int error_count = 0;
  std::vector<std::vector<float>> reference_result =
      engine.RunInference(input_data);
  for (int r = 0; r < runs; ++r) {
    VLOG_EVERY_N(0, 100) << "inference running iter " << r << "...";
    const auto& result = engine.RunInference(input_data);
    const int num_outputs = result.size();
    CHECK_GT(num_outputs, 0);
    for (int i = 0; i < num_outputs; ++i) {
      for (int j = 0; j < result[i].size(); ++j) {
        if (result[i][j] != reference_result[i][j]) {
          VLOG(1) << "[ iteration = " << r << " ] output of tensor " << i
                  << " at position " << j << " differs from reference.\n"
                  << "( output = " << result[i][j]
                  << " reference = " << reference_result[i][j] << " )";
          ++error_count;
        }
      }
    }
  }
  EXPECT_EQ(0, error_count) << "total runs " << runs;
}

void InferenceStressTest(const std::string& model_path, int runs,
                         int sleep_sec) {
  BasicEngine engine(model_path);
  for (int i = 0; i < runs; ++i) {
    VLOG_EVERY_N(0, 100) << "inference running iter " << i << "...";
    const auto& input_data = GetRandomInput(engine.get_input_tensor_shape());
    const auto& result = engine.RunInference(input_data);
    CHECK(!result.empty());
    sleep(sleep_sec);
  }
}

}  // namespace coral
