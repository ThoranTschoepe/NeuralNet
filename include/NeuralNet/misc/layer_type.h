#pragma once

namespace NeuralNet {

  enum LayerType {
    FullyConnected = 0,
    ReLU = 1000,
    LeakyReLU = 1001,
    Sigmoid = 1002,
    Softmax = 1003,
    Tanh = 1004,

    DEFAULT = 9999
  };

  template<typename T>
  struct LayerTypeTraits {
    static constexpr LayerType type = LayerType::DEFAULT;
  };

}