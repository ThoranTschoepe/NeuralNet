#pragma once

#include <NeuralNet/misc/types.h>

namespace NeuralNet::Training {

  template<std::floating_point dataType = NNFLOAT>
  class NesterovOptimizer : public BaseOptimizer<dataType> {
   private:
    dataType learning_rate_;
    dataType momentum_;
    std::vector<std::vector<dataType>> cache_;
    std::vector<std::vector<dataType>> old_cache_;
   public:
    explicit NesterovOptimizer(dataType learning_rate = 0.01, dataType momentum = 0.9)
        : learning_rate_(learning_rate), momentum_(momentum) {}

    void Allocate(const std::shared_ptr<const NeuralNetwork<dataType>> net) override {
      for (const auto &layer : *net) {
        cache_.emplace_back(layer->ParametersSize());
        old_cache_.emplace_back(layer->ParametersSize());
      }
    }

    void CalculateUpdatesFromGradients(std::vector<dataType> &updates, size_t layer_id) override {
      size_t update_size = updates.size();

      for (size_t i = 0; i < update_size; ++i) {
        old_cache_[layer_id][i] = cache_[layer_id][i];
        cache_[layer_id][i] = momentum_ * cache_[layer_id][i] - learning_rate_ * updates[i];
        updates[i] = - momentum_ * old_cache_[layer_id][i] + (1 + momentum_) * cache_[layer_id][i];
      }
    }
  };

}