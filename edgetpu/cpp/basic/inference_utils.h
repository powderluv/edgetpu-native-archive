// This library contains utility functions to run DNN inference on EdgeTpu.

#ifndef EDGETPU_CPP_BASIC_INFERENCE_UTILS_H_
#define EDGETPU_CPP_BASIC_INFERENCE_UTILS_H_

#include <array>
#include <cstdint>
#include <vector>

#include "absl/base/macros.h"
#include "edgetpu.h"
#include "edgetpu/cpp/error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

namespace coral {

// Defines dimension of an image, in height, width, depth order.
typedef std::array<int, 3> ImageDims;

// The box is represented by a vector with 4 coordinates: x1, y1, x2, y2.
// First point is left-top while second is right-bottom.
typedef std::array<float, 4> Box;

// Returns whether string ends with given suffix.
inline bool EndsWith(std::string const& value, std::string const& ending) {
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

// Compute IoU between two boxes.
float IntersectionOverUnion(const Box& box1, const Box& box2);

// Returns total number of elements.
int ImageDimsToSize(const ImageDims& dims);

// Reads BMP image. Returns empty vector upon failure.
std::vector<uint8_t> ReadBmp(const std::string& input_bmp_name,
                             ImageDims* image_dims);

// Resizes BMP image.
void ResizeImage(const ImageDims& in_dims, const uint8_t* in,
                 const ImageDims& out_dims, uint8_t* out);

// Converts RGB image to grayscale. Take the average.
std::vector<uint8_t> RgbToGrayscale(const std::vector<uint8_t>& in,
                                    const ImageDims& in_dims);

// Gets input from images and resizes to `target_dims`. Returns empty vector
// upon failure.
std::vector<uint8_t> GetInputFromImage(const std::string& image_path,
                                       const ImageDims& target_dims);

// Returns the output tensor sizes of the given model, assuming all tensors have
// been allocated.
std::vector<int> GetOutputTensorSizes(const tflite::Interpreter& interpreter);

// Runs inference with CPU or EdgeTpu tflite model, and returns raw inference
// results. It assumes memory of input and output tensors have been allocated
// properly. Note that output is concatenated list of all output tensor values.
bool RunInferenceHelper(const uint8_t* const input_data, int input_size,
                        int output_size, tflite::Interpreter* interpreter,
                        float* output);

// Builds interpreter from model that can run on EdgeTpu, and allocates tensors.
std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(
    const tflite::FlatBufferModel& model,
    tflite::ops::builtin::BuiltinOpResolver* resolver,
    edgetpu::EdgeTpuContext* edgetpu_context,
    EdgeTpuErrorReporter* error_reporter);

}  // namespace coral

#endif  // EDGETPU_CPP_BASIC_INFERENCE_UTILS_H_
