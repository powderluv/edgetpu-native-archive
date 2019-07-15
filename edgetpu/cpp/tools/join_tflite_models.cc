// Tool to join two tflite models. The models may contain custom operator,
// which can not be imported / exported properly by tflite/toco yet.
//
// Two models can be joined together only if the output tensors of
// input_graph_base are the input tensors of input_graph_head. All other tensors
// should have identical names.

#include "edgetpu/cpp/tools/tflite_graph_util.h"
#include "gflags/gflags.h"

DEFINE_string(input_graph_base, "",
              "Path to the base input graph. Must be in tflite format.");

DEFINE_string(input_graph_head, "",
              "Path to the head input graph. Must be in tflite format.");

DEFINE_string(
    output_graph, "",
    "Path to the output graph. Output graph will be in tflite format.");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  coral::tools::ConcatTfliteModels(FLAGS_input_graph_base,
                                   FLAGS_input_graph_head, FLAGS_output_graph);
}
