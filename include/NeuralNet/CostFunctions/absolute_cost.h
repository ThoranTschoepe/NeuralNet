#pragma once

#include <NeuralNet/CostFunctions/base_cost.h>

namespace NeuralNet::Training {

  template<std::floating_point dataType = NNFLOAT>
  class AbsoluteCost : public BaseCost<dataType> {
   public:

    [[nodiscard]] dataType
    Cost(const std::vector<dataType> &output, const std::vector<dataType> &target) const override {
      assert(output.size() == target.size());

      size_t output_size = output.size();
      dataType cost = 0;

      for (size_t i = 0; i < output_size; ++i) {
        cost += std::abs(output[i] - target[i]);
      }

      cost /= static_cast<dataType>(output_size);

      return cost;
    }

    [[nodiscard]] std::vector<dataType>
    Gradient(const std::vector<dataType> &output, const std::vector<dataType> &target) const override {
      assert(output.size() == target.size());

      size_t output_size = output.size();
      std::vector<dataType> grad(output_size);
      dataType diff;

      for (size_t i = 0; i < output_size; ++i) {
        diff = output[i] - target[i];

        if (diff > 0) {
          grad[i] = 1;
        } else if (diff < 0) {
          grad[i] = -1;
        } else {
          grad[i] = 0;
        }
      }

      return grad;
    }
  };

}