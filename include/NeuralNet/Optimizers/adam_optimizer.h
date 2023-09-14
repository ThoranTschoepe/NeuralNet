#pragma once

#include <NeuralNet/Optimizers/base_optimizer.h>

namespace NeuralNet::Training {

  template<std::floating_point dataType = NNFLOAT>
  class AdamOptimizer : public BaseOptimizer<dataType> {
   private:

    dataType beta1_;
    dataType beta2_;
    dataType epsilon_;
    dataType learning_rate_;

    std::vector<std::vector<dataType>> m_;
    std::vector<std::vector<dataType>> v_;

    std::vector<std::vector<dataType>> b_t_;

   public:

    explicit AdamOptimizer(dataType learning_rate = 0.01,
                           dataType beta1 = 0.9,
                           dataType beta2 = 0.999,
                           dataType epsilon = 1e-8)
        : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {}

    void Allocate(const std::shared_ptr<const NeuralNetwork<dataType>> net) override {
      for (const auto &layer : *net) {
        m_.emplace_back(layer->ParametersSize());
        v_.emplace_back(layer->ParametersSize());

        b_t_.emplace_back(std::vector<dataType>{beta1_, beta2_});
      }
    }

    void CalculateUpdatesFromGradients(std::vector<dataType> &updates, size_t layer_id) override {
      size_t updates_size = updates.size();

      for (size_t i = 0; i < updates_size; ++i) {
        m_[layer_id][i] = beta1_ * m_[layer_id][i] + (1 - beta1_) * updates[i];
        v_[layer_id][i] = beta2_ * v_[layer_id][i] + (1 - beta2_) * std::pow(updates[i], 2);

        dataType m_hat = m_[layer_id][i] / (1 - b_t_[layer_id][0]);
        dataType v_hat = v_[layer_id][i] / (1 - b_t_[layer_id][1]);

        updates[i] = -learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
      }

      b_t_[layer_id][0] *= beta1_;
      b_t_[layer_id][1] *= beta2_;
    }
  };

}