#pragma once

#include <NeuralNet/CostFunctions/base_cost.h>

namespace NeuralNet::Training {

  template<std::floating_point dataType = NNFLOAT>
  class CrossEntropyCost : public BaseCost<dataType> {
   public:

    [[nodiscard]] dataType
    Cost(const std::vector<dataType> &output, const std::vector<dataType> &target) const override {
      assert(output.size() == target.size());

      size_t output_size = output.size();
      dataType cost = 0;

      for (size_t i = 0; i < output_size; ++i) {
        cost += target[i] * std::log(output[i]) + (1 - target[i]) * std::log(1 - output[i]);
      }

      return -cost;
    }

    [[nodiscard]] std::vector<dataType>
    Gradient(const std::vector<dataType> &output, const std::vector<dataType> &target) const override {
      assert(output.size() == target.size());

      size_t output_size = output.size();
      std::vector<dataType> grad(output_size);

      for (size_t i = 0; i < output_size; ++i) {
        grad[i] = (output[i] - target[i]) / (output[i] * (1 - output[i]));
      }

      return grad;
    }
  };

}
