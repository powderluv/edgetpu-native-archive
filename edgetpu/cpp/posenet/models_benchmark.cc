#include "benchmark/benchmark.h"
#include "edgetpu/cpp/test_utils.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

namespace coral {

DEFINE_string(posenet_model_dir,
              "knowledge/cerebra/spacepark/darwinn/git/edgetpu/cpp/"
              "posenet/test_data",
              "Posenet model directory");

std::string PosenetModelPath(const std::string& model_name) {
  return std::string(FLAGS_posenet_model_dir) + "/" + model_name;
}

template <CnnProcessorType CnnProcessor, int ysize, int xsize>
static void BM_PoseNet_MobileNetV1_075_WithDecoder(benchmark::State& state) {
  const std::string model_prefix = "posenet_mobilenet_v1_075_" +
                                   std::to_string(ysize) + "_" +
                                   std::to_string(xsize);
  const std::string model_path =
      PosenetModelPath((CnnProcessor == kEdgeTpu)
                           ? model_prefix + "_quant_decoder_edgetpu.tflite"
                           : model_prefix + "_quant_decoder.tflite");
  coral::BenchmarkModelOnEdgeTpu(model_path, state);
}
BENCHMARK_TEMPLATE(BM_PoseNet_MobileNetV1_075_WithDecoder, kEdgeTpu, 353, 481);
BENCHMARK_TEMPLATE(BM_PoseNet_MobileNetV1_075_WithDecoder, kCpu, 353, 481);
BENCHMARK_TEMPLATE(BM_PoseNet_MobileNetV1_075_WithDecoder, kEdgeTpu, 481, 641);
BENCHMARK_TEMPLATE(BM_PoseNet_MobileNetV1_075_WithDecoder, kCpu, 481, 641);
BENCHMARK_TEMPLATE(BM_PoseNet_MobileNetV1_075_WithDecoder, kEdgeTpu, 721, 1281);
BENCHMARK_TEMPLATE(BM_PoseNet_MobileNetV1_075_WithDecoder, kCpu, 721, 1281);

}  // namespace coral

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
