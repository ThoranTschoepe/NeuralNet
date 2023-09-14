#pragma once

#include <NeuralNet/CostFunctions/base_cost.h>

namespace NeuralNet::Training {

  template<std::floating_point dataType = NNFLOAT>
  class MSECost : public BaseCost<dataType> {
   public:

    [[nodiscard]] dataType
    Cost(const std::vector<dataType> &output, const std::vector<dataType> &target) const override {
      assert(output.size() == target.size());

      size_t output_size = output.size();
      dataType cost = 0;
      dataType diff;

      for (size_t i = 0; i < output_size; ++i) {
        diff = output[i] - target[i];
        cost += diff * diff;
      }

      return cost / static_cast<dataType>(2 * output_size);
    }

    [[nodiscard]] std::vector<dataType>
    Gradient(const std::vector<dataType> &output, const std::vector<dataType> &target) const override {
      assert(output.size() == target.size());

      size_t output_size = output.size();
      std::vector<dataType> grad(output_size);

      for (size_t i = 0; i < output_size; ++i) {
        grad[i] = output[i] - target[i];
      }

      return grad;
    }
  };

}
