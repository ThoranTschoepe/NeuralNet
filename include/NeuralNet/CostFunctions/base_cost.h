#pragma once

#include <NeuralNet/misc/types.h>

namespace NeuralNet::Training {

  template<std::floating_point dataType>
  class BaseCost {
   public:

    [[nodiscard]] virtual dataType
    Cost(const std::vector<dataType> &output, const std::vector<dataType> &target) const = 0;

    [[nodiscard]] virtual std::vector<dataType>
    Gradient(const std::vector<dataType> &output, const std::vector<dataType> &target) const = 0;

  };

}