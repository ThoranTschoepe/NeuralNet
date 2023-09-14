#pragma once

#include <NeuralNet/Layers/base_layer.h>

#include <random>

namespace NeuralNet {

  template<std::floating_point dataType>
  class BaseTrainableLayer : public BaseLayer<dataType> {
   public:

    BaseTrainableLayer(size_t input_size, size_t output_size) : BaseLayer<dataType>(input_size, output_size) {}

    [[nodiscard]] bool Trainable() const override {
      return true;
    }

    [[nodiscard]] virtual std::vector<dataType> &Parameters() = 0;

    virtual void UpdateParameters(const std::vector<dataType> &updates) = 0;
  };

}