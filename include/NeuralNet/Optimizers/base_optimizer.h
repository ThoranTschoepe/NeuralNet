#pragma once

#include <NeuralNet/misc/types.h>

namespace NeuralNet::Training {

  template<std::floating_point dataType>
  class BaseOptimizer {
   public:

    virtual void Allocate(std::shared_ptr<const NeuralNetwork<dataType>> net) = 0;

    virtual void CalculateUpdatesFromGradients(std::vector<dataType> &updates, size_t layer_id) = 0;
  };

}