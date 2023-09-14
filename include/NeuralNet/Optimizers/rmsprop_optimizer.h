#pragma once

#include <NeuralNet/Optimizers/base_optimizer.h>

namespace NeuralNet::Training {

  template<std::floating_point dataType = NNFLOAT>
  class RMSPropOptimizer : public BaseOptimizer<dataType> {
   private:

    dataType learning_rate_;
    dataType decay_rate_;
    dataType epsilon_;

    std::vector<std::vector<dataType>> cache_;

   public:

    explicit RMSPropOptimizer(dataType learning_rate = 0.01, dataType decay_rate = 0.9, dataType epsilon = 1e-8)
        : learning_rate_(learning_rate), decay_rate_(decay_rate), epsilon_(epsilon) {}

    void Allocate(const std::shared_ptr<const NeuralNetwork<dataType>> net) override {
      for (const auto &layer : *net) {
        cache_.emplace_back(layer->ParametersSize());
      }
    }

    void CalculateUpdatesFromGradients(std::vector<dataType> &updates, size_t layer_id) override {
      size_t update_size = updates.size();

      for (size_t i = 0; i < update_size; ++i) {
        cache_[layer_id][i] = decay_rate_ * cache_[layer_id][i] + (1 - decay_rate_) * updates[i] * updates[i];
        updates[i] = - learning_rate_ * updates[i] / (std::sqrt(cache_[layer_id][i]) + epsilon_);
      }
    }
  };

}