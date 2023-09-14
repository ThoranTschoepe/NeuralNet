#pragma once

#include <NeuralNet/Layers/Activations/base_activation.h>
#include <NeuralNet/misc/math_util.h>

namespace NeuralNet {

  template<std::floating_point dataType = NNFLOAT>
  class SigmoidActivation;

  template<std::floating_point T>
  struct LayerTypeTraits<SigmoidActivation<T>> {
    static constexpr LayerType type = LayerType::Sigmoid;
  };

  template<std::floating_point dataType>
  class SigmoidActivation : public BaseActivation<dataType> {
   public:

    SigmoidActivation(size_t input_size, size_t output_size) : BaseActivation<dataType>(input_size, output_size) {}

    void ForwardActivate(const std::vector<dataType> &input, std::vector<dataType> &output) override {
      size_t input_size = input.size();
      for (size_t i = 0; i < input_size; ++i) {
        output[i] = 1 / (1 + exp(-input[i]));
      }
    }

    void BackwardActivate(const std::vector<dataType> &input, const std::vector<dataType> &output,
                          const std::vector<dataType> &delta, std::vector<dataType> &prev_delta) override {
      size_t input_size = input.size();
      for (size_t i = 0; i < input_size; ++i) {
        prev_delta[i] = delta[i] * output[i] * (1 - output[i]);
      }
    }

    void Print(std::ostream &os, bool) const override {
      os << "ID: " << this->layer_id_ << " Sigmoid activation: " << this->input_size_ << " -> "
         << this->output_size_ << std::endl;
    }

    void Save(std::ofstream &file) const override {
      file.write(reinterpret_cast<const char *>(&LayerTypeTraits<SigmoidActivation<dataType>>::type),
                 sizeof(LayerType));
      file.write(reinterpret_cast<const char *>(&this->input_size_), sizeof(size_t));
    }
  };

}