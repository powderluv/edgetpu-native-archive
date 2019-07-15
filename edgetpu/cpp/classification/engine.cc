#include "edgetpu/cpp/classification/engine.h"

#include <algorithm>
#include <functional>
#include <queue>

#include "glog/logging.h"

namespace coral {

void ClassificationEngine::Validate() {
  std::vector<int> output_tensor_sizes = get_all_output_tensors_sizes();
  CHECK_EQ(output_tensor_sizes.size(), 1)
      << "Format error: classification model should have one output tensor "
         "only!";
}

std::vector<ClassificationCandidate>
ClassificationEngine::ClassifyWithInputTensor(const std::vector<uint8_t>& input,
                                              float threshold, int top_k) {
  std::vector<float> scores = RunInference(input)[0];
  std::priority_queue<ClassificationCandidate,
                      std::vector<ClassificationCandidate>,
                      std::greater<ClassificationCandidate>>
      q;
  for (int i = 0; i < scores.size(); ++i)
    if (scores[i] > threshold) {
      ClassificationCandidate tmp(i, scores[i]);
      if (q.size() < top_k || tmp > q.top()) {
        if (q.size() >= top_k) {
          // Remove the smallest one.
          q.pop();
        }
        q.push(tmp);
      }
    }

  std::vector<ClassificationCandidate> ret;
  while (!q.empty()) {
    ret.push_back(q.top());
    q.pop();
  }
  std::reverse(ret.begin(), ret.end());
  return ret;
}

}  // namespace coral
