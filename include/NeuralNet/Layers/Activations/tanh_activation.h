#pragma once

#include <NeuralNet/Layers/Activations/base_activation.h>

namespace NeuralNet {

  template<std::floating_point dataType = NNFLOAT>
  class TanhActivation;

  template<std::floating_point T>
  struct LayerTypeTraits<TanhActivation<T>> {
    static constexpr LayerType type = LayerType::Tanh;
  };

  template<std::floating_point dataType>
  class TanhActivation : public BaseActivation<dataType> {
   public:

    TanhActivation(size_t input_size, size_t output_size) : BaseActivation<dataType>(input_size, output_size) {}

    void ForwardActivate(const std::vector<dataType> &input, std::vector<dataType> &output) override {
      size_t input_size = input.size();
      for (size_t i = 0; i < input_size; ++i) {
        output[i] = std::tanh(input[i]);
      }
    }

    void BackwardActivate(const std::vector<dataType> &input, const std::vector<dataType> &output,
                          const std::vector<dataType> &delta, std::vector<dataType> &prev_delta) override {
      size_t input_size = input.size();
      for (size_t i = 0; i < input_size; ++i) {
        prev_delta[i] = delta[i] * (dataType(1) - output[i] * output[i]);
      }
    }

    void Print(std::ostream &os, bool) const override {
      os << "ID: " << this->layer_id_ << " Tanh activation: " << this->input_size_ << " -> " << this->output_size_
         << std::endl;
    }

    void Save(std::ofstream &file) const override {
      file.write(reinterpret_cast<const char *>(&LayerTypeTraits<TanhActivation<dataType>>::type), sizeof(LayerType));
      file.write(reinterpret_cast<const char *>(&this->input_size_), sizeof(size_t));
    }
  };

}