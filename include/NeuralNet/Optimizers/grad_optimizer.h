#pragma once

#include <NeuralNet/Optimizers/base_optimizer.h>

namespace NeuralNet::Training {

  template<std::floating_point dataType = NNFLOAT>
  class GradOptimizer : public BaseOptimizer<dataType> {
    dataType learning_rate_;

   public:

    explicit GradOptimizer(dataType learning_rate = 0.01) : learning_rate_(learning_rate) {}

    void Allocate(const std::shared_ptr<const NeuralNetwork<dataType>>) override {}

    void CalculateUpdatesFromGradients(std::vector<dataType> &updates, size_t) override {
      size_t update_size = updates.size();

      for (size_t i = 0; i < update_size; ++i) {
        updates[i] *= -learning_rate_;
      }
    }
  };

}