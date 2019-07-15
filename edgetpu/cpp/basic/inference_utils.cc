#include "edgetpu/cpp/basic/inference_utils.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "edgetpu/cpp/posenet/posenet_decoder_op.h"
#include "edgetpu/cpp/utils.h"
#include "glog/logging.h"
#include "tensorflow/lite/builtin_op_data.h"

namespace coral {
namespace {

using tflite::ops::builtin::BuiltinOpResolver;

std::vector<uint8_t> _DecodeBmp(const uint8_t* input, int row_size, int width,
                                int height, int channels, bool top_down) {
  std::vector<uint8_t> output(height * width * channels);
  for (int i = 0; i < height; i++) {
    int src_pos;
    int dst_pos;
    for (int j = 0; j < width; j++) {
      if (!top_down) {
        src_pos = ((height - 1 - i) * row_size) + j * channels;
      } else {
        src_pos = i * row_size + j * channels;
      }
      dst_pos = (i * width + j) * channels;
      switch (channels) {
        case 1:
          output[dst_pos] = input[src_pos];
          break;
        case 3:
          // BGR -> RGB
          output[dst_pos] = input[src_pos + 2];
          output[dst_pos + 1] = input[src_pos + 1];
          output[dst_pos + 2] = input[src_pos];
          break;
        case 4:
          // BGRA -> RGBA
          output[dst_pos] = input[src_pos + 2];
          output[dst_pos + 1] = input[src_pos + 1];
          output[dst_pos + 2] = input[src_pos];
          output[dst_pos + 3] = input[src_pos + 3];
          break;
        default:
          LOG(FATAL) << "Unexpected number of channels: " << channels;
          break;
      }
    }
  }
  return output;
}

inline float _Area(const Box& box) {
  return (box[2] - box[0]) * (box[3] - box[1]);
}

}  // namespace

float IntersectionOverUnion(const Box& box1, const Box& box2) {
  const Box intersect = {std::max(box1[0], box2[0]), std::max(box1[1], box2[1]),
                         std::min(box1[2], box2[2]),
                         std::min(box1[3], box2[3])};
  // No overlap.
  if (_Area(intersect) < 0) return 0.0;
  return _Area(intersect) / (_Area(box1) + _Area(box2) - _Area(intersect));
}

int ImageDimsToSize(const ImageDims& dims) {
  int size = 1;
  for (const auto& dim : dims) {
    size *= dim;
  }
  return size;
}

void ResizeImage(const ImageDims& in_dims, const uint8_t* in,
                 const ImageDims& out_dims, uint8_t* out) {
  const int image_height = in_dims[0];
  const int image_width = in_dims[1];
  const int image_channels = in_dims[2];
  const int wanted_height = out_dims[0];
  const int wanted_width = out_dims[1];
  const int wanted_channels = out_dims[2];
  const int number_of_pixels = image_height * image_width * image_channels;
  std::unique_ptr<tflite::Interpreter> interpreter(new tflite::Interpreter);
  int base_index = 0;
  // two inputs: input and new_sizes
  interpreter->AddTensors(2, &base_index);
  // one output
  interpreter->AddTensors(1, &base_index);
  // set input and output tensors
  interpreter->SetInputs({0, 1});
  interpreter->SetOutputs({2});
  // set parameters of tensors
  TfLiteQuantizationParams quant;
  interpreter->SetTensorParametersReadWrite(
      0, kTfLiteFloat32, "input",
      {1, image_height, image_width, image_channels}, quant);
  interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", {2},
                                            quant);
  interpreter->SetTensorParametersReadWrite(
      2, kTfLiteFloat32, "output",
      {1, wanted_height, wanted_width, wanted_channels}, quant);
  BuiltinOpResolver resolver;
  const TfLiteRegistration* resize_op =
      resolver.FindOp(tflite::BuiltinOperator_RESIZE_BILINEAR, 1);
  auto* params = reinterpret_cast<TfLiteResizeBilinearParams*>(
      malloc(sizeof(TfLiteResizeBilinearParams)));
  params->align_corners = false;
  interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params, resize_op,
                                     nullptr);
  interpreter->AllocateTensors();
  // fill input image
  // in[] are integers, cannot do memcpy() directly
  auto input = interpreter->typed_tensor<float>(0);
  for (int i = 0; i < number_of_pixels; i++) {
    input[i] = in[i];
  }
  // fill new_sizes
  interpreter->typed_tensor<int>(1)[0] = wanted_height;
  interpreter->typed_tensor<int>(1)[1] = wanted_width;
  interpreter->Invoke();
  auto output = interpreter->typed_tensor<float>(2);
  auto output_number_of_pixels =
      wanted_height * wanted_height * wanted_channels;
  for (int i = 0; i < output_number_of_pixels; i++) {
    out[i] = static_cast<uint8_t>(output[i]);
  }
}

std::vector<uint8_t> ReadFileContents(const std::string& file_name) {
  int begin, end;
  std::ifstream file(file_name, std::ios::in | std::ios::binary);
  if (!file) return {};

  begin = file.tellg();
  file.seekg(0, std::ios::end);
  end = file.tellg();
  size_t len = end - begin;
  VLOG(1) << "len: " << len << "\n";
  std::vector<uint8_t> file_bytes(len);
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char*>(file_bytes.data()), len);
  return file_bytes;
}

std::vector<uint8_t> ReadBmp(const std::string& input_bmp_name,
                             ImageDims* image_dims) {
  std::string file_content;
  ReadFileOrDie(input_bmp_name, &file_content);
  if (file_content.empty()) return {};
  const uint8_t* img_bytes =
      reinterpret_cast<const uint8_t*>(file_content.data());

  // Data in BMP file header is stored in Little Endian format. The following
  // method should work on both Big and Little Endian machine.
  auto to_int32 = [](const unsigned char* p) -> int32_t {
    return p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
  };
  const int32_t header_size = to_int32(img_bytes + 10);
  const int32_t bpp = to_int32(img_bytes + 28);
  int* width = image_dims->data() + 1;
  int* height = image_dims->data();
  int* channels = image_dims->data() + 2;
  *width = to_int32(img_bytes + 18);
  *height = to_int32(img_bytes + 22);
  *channels = bpp / 8;
  // Currently supports RGB and grayscale image at this function.
  if ((*width) < 0 || (*height) < 0 || ((*channels) != 3 && (*channels) != 1)) {
    return {};
  }
  VLOG(1) << "width, height, channels: " << *width << ", " << *height << ", "
          << *channels << "\n";

  // there may be padding bytes when the width is not a multiple of 4 bytes
  // 8 * channels == bits per pixel
  const int row_size = (8 * (*channels) * (*width) + 31) / 32 * 4;
  // if height is negative, data layout is top down
  // otherwise, it's bottom up
  bool top_down = (*height < 0);
  // Decode image, allocating tensor once the image size is known
  const uint8_t* bmp_pixels = &img_bytes[header_size];
  return _DecodeBmp(bmp_pixels, row_size, *width, abs(*height), *channels,
                    top_down);
}

std::vector<uint8_t> RgbToGrayscale(const std::vector<uint8_t>& in,
                                    const ImageDims& in_dims) {
  CHECK_GE(in_dims[2], 3);
  std::vector<uint8_t> result;
  int out_size = in_dims[0] * in_dims[1];
  result.resize(out_size);
  for (int in_idx = 0, out_idx = 0; in_idx < in.size();
       in_idx += in_dims[2], ++out_idx) {
    int r = in[in_idx];
    int g = in[in_idx + 1];
    int b = in[in_idx + 2];
    result[out_idx] = static_cast<uint8_t>((r + g + b) / 3);
  }
  return result;
}

std::vector<uint8_t> GetInputFromImage(const std::string& image_path,
                                       const ImageDims& target_dims) {
  std::vector<uint8_t> result;
  if (!EndsWith(image_path, ".bmp")) {
    LOG(FATAL) << "Unsupported image type: " << image_path;
    return result;
  }
  result.resize(ImageDimsToSize(target_dims));
  ImageDims image_dims;
  std::vector<uint8_t> in = ReadBmp(image_path, &image_dims);
  if (in.empty()) return {};
  if (target_dims[2] == 1 && (image_dims[2] == 3 || image_dims[2] == 4)) {
    in = RgbToGrayscale(in, image_dims);
  }
  ResizeImage(image_dims, in.data(), target_dims, result.data());
  return result;
}

std::vector<int> GetOutputTensorSizes(const tflite::Interpreter& interpreter) {
  std::vector<int> output_tensor_sizes;
  const std::vector<int>& indices = interpreter.outputs();
  output_tensor_sizes.resize(indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    const auto* output_tensor = interpreter.tensor(indices[i]);
    if (output_tensor->type == kTfLiteUInt8) {
      output_tensor_sizes[i] = output_tensor->bytes;
    } else if (output_tensor->type == kTfLiteFloat32) {
      output_tensor_sizes[i] = output_tensor->bytes / sizeof(float);
    } else if (output_tensor->type == kTfLiteInt64) {
      output_tensor_sizes[i] = output_tensor->bytes / sizeof(int64_t);
    } else {
      LOG(FATAL) << "output tensor type not supported! output_tensor->type = "
                 << output_tensor->type;
    }
  }
  return output_tensor_sizes;
}

bool RunInferenceHelper(const uint8_t* const input_data, int input_size,
                        int output_size, tflite::Interpreter* interpreter,
                        float* output_data) {
  CHECK(input_data);
  CHECK(output_data);
  uint8_t* input = interpreter->typed_input_tensor<uint8_t>(0);
  CHECK(input);
  std::memcpy(input, input_data, input_size);
  if (interpreter->Invoke() != kTfLiteOk) {
    return false;
  }

  const auto& output_indices = interpreter->outputs();
  const int num_outputs = output_indices.size();
  int out_idx = 0;
  for (int i = 0; i < num_outputs; ++i) {
    const auto* out_tensor = interpreter->tensor(output_indices[i]);
    CHECK(out_tensor);
    if (out_tensor->type == kTfLiteUInt8) {
      const int num_values = out_tensor->bytes;
      const uint8_t* output = interpreter->typed_output_tensor<uint8_t>(i);
      CHECK(output);
      for (int j = 0; j < num_values; ++j) {
        output_data[out_idx++] = (output[j] - out_tensor->params.zero_point) *
                                 out_tensor->params.scale;
      }
    } else if (out_tensor->type == kTfLiteFloat32) {
      const int num_values = out_tensor->bytes / sizeof(float);
      const float* output = interpreter->typed_output_tensor<float>(i);
      CHECK(output);
      for (int j = 0; j < num_values; ++j) {
        output_data[out_idx++] = output[j];
      }
    } else if (out_tensor->type == kTfLiteInt64) {
      const int num_values = out_tensor->bytes / sizeof(int64_t);
      const int64_t* output = interpreter->typed_output_tensor<int64_t>(i);
      CHECK(output);
      for (int j = 0; j < num_values; ++j) {
        output_data[out_idx++] = output[j];
      }
    } else {
      LOG(FATAL) << "Tensor " << out_tensor->name
                 << " has unsupported output type: " << out_tensor->type;
    }
    CHECK_LE(out_idx, output_size);
  }
  CHECK_EQ(out_idx, output_size);
  return true;
}

std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(
    const tflite::FlatBufferModel& model, BuiltinOpResolver* resolver,
    edgetpu::EdgeTpuContext* edgetpu_context,
    EdgeTpuErrorReporter* error_reporter) {
  if (resolver == nullptr) {
    error_reporter->Report("nullptr resolver.");
    return nullptr;
  }
  resolver->AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  resolver->AddCustom(kPosenetDecoderOp, RegisterPosenetDecoderOp());
  std::unique_ptr<tflite::Interpreter> interpreter;
  // When BasicEngine is initializing with FlatBufferModel, it's possible that
  // there is no ErrorReporter binded with it.
  tflite::InterpreterBuilder interpreter_builder(model.GetModel(), *resolver,
                                                 error_reporter);
  if (interpreter_builder(&interpreter) != kTfLiteOk) {
    error_reporter->Report("Error in interpreter initialization.");
    return nullptr;
  }
  // Bind given context with interpreter.
  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
  interpreter->SetNumThreads(1);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("Failed to allocate tensors.");
    return nullptr;
  }
  return interpreter;
}

}  // namespace coral
