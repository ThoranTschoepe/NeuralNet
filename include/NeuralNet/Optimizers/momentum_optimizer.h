#pragma once

#include <NeuralNet/Optimizers/base_optimizer.h>

namespace NeuralNet::Training {

  template<std::floating_point dataType = NNFLOAT>
  class MomentumOptimizer : public BaseOptimizer<dataType> {
   private:

    dataType learning_rate_;
    dataType momentum_;
    std::vector<std::vector<dataType>> cache_;

   public:

    explicit MomentumOptimizer(dataType learning_rate = 0.01, dataType momentum = 0.9)
        : learning_rate_(learning_rate), momentum_(momentum) {}

    void Allocate(const std::shared_ptr<const NeuralNetwork<dataType>> net) override {
      for (const auto &layer : *net) {
        cache_.emplace_back(layer->ParametersSize());
      }
    }

    void CalculateUpdatesFromGradients(std::vector<dataType> &updates, size_t layer_id) override {
      size_t update_size = updates.size();

      for (size_t i = 0; i < update_size; ++i) {
        cache_[layer_id][i] = momentum_ * cache_[layer_id][i] - learning_rate_ * updates[i];
        updates[i] = cache_[layer_id][i];
      }
    }
  };

}