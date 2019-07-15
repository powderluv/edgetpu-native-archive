#ifndef EDGETPU_CPP_TEST_UTILS_H_
#define EDGETPU_CPP_TEST_UTILS_H_

#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "edgetpu/cpp/basic/inference_utils.h"

namespace coral {

enum CnnProcessorType { kEdgeTpu, kCpu };

enum CompilationType { kCoCompilation, kSingleCompilation };

// Retrieves model path with model name (file name).
std::string ModelPath(const std::string& model_name);

// Retrieves test file path with file name.
std::string TestDataPath(const std::string& name);

// Generates a 1-d uint8 array with given size.
std::vector<uint8_t> GetRandomInput(int n);

// Generates a 1-d uint8 array with given input tensor shape.
std::vector<uint8_t> GetRandomInput(std::vector<int> shape);

// Gets list of all models.
std::vector<std::string> GetAllModels();

// Tests model with random input. Ensures it's runnable.
void TestWithRandomInput(const std::string& model_path);

// Tests model with a real image.
std::vector<std::vector<float>> TestWithImage(const std::string& model_path,
                                              const std::string& image_path);

// Returns top-k predictions as label-score pairs.
std::vector<std::pair<int, float>> GetTopK(const std::vector<float>& scores,
                                           float threshold, int top_k);

// Returns whether top k results contains a given label.
bool TopKContains(const std::vector<std::pair<int, float>>& topk, int label);

// Tests a classification model with customized preprocessing.
// Custom preprocessing is done by:
// (v - (mean - zero_point * scale * stddev)) / (stddev * scale)
// where zero_point and scale are the quantization parameters of the input
// tensor, and mean and stddev are the normalization parameters of the input
// tensor. Effective mean and scale should be
// (mean - zero_point * scale * stddev) and (stddev * scale) respectively.
void TestClassification(const std::string& model_path,
                        const std::string& image_path, float effective_scale,
                        const std::vector<float>& effective_means,
                        float score_threshold, int k, int expected_topk_label);

// Tests a classification model.
void TestClassification(const std::string& model_path,
                        const std::string& image_path, float score_threshold,
                        int k, int expected_topk_label);

// Tests a classification model. Only checks the top1 result.
void TestClassification(const std::string& model_path,
                        const std::string& image_path, float score_threshold,
                        int expected_top1_label);

// Tests a SSD detection model. Only checks the first detection result.
void TestDetection(const std::string& model_path, const std::string& image_path,
                   const Box& expected_box, int expected_label,
                   float score_threshold, float iou_threshold);

// Tests a MSCOCO detection model with cat.bmp.
void TestCatMsCocoDetection(const std::string& model_path,
                            float score_threshold, float iou_threshold);

void BenchmarkModelOnEdgeTpu(const std::string& model_path,
                             benchmark::State& state);

// Benchmarks models on a sinlge EdgeTpu device.
void BenchmarkModelsOnEdgeTpu(const std::vector<std::string>& model_paths,
                              benchmark::State& state);

// This test will run inference with fixed randomly generated input for multiple
// times and ensure the inference result are constant.
//  - model_path: string, path to the FlatBuffer file.
//  - runs: number of iterations.
void RepeatabilityTest(const std::string& model_path, int runs);

// This test will run inference with given model for multiple times. Input are
// generated randomly and the result won't be checked.
//  - model_path: string, path of the FlatBuffer file.
//  - runs: number of iterations.
//  - sleep_sec: time interval between inferences. By default it's zero.
void InferenceStressTest(const std::string& model_path, int runs,
                         int sleep_sec = 0);

}  // namespace coral

#endif  // EDGETPU_CPP_TEST_UTILS_H_
