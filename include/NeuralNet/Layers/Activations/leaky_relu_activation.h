#pragma once

#include <NeuralNet/Layers/Activations/base_activation.h>

namespace NeuralNet {

  template<std::floating_point dataType = NNFLOAT>
  class LeakyReLuActivation;

  template<std::floating_point T>
  struct LayerTypeTraits<LeakyReLuActivation<T>> {
    static constexpr LayerType type = LayerType::LeakyReLU;
  };

  template<std::floating_point dataType>
  class LeakyReLuActivation : public BaseActivation<dataType> {
   private:

    dataType alpha_;

   public:

    explicit LeakyReLuActivation(size_t input_size, size_t output_size, float alpha = 0.01f)
        : BaseActivation<dataType>(input_size, output_size), alpha_(alpha) {}

    void ForwardActivate(const std::vector<dataType> &input, std::vector<dataType> &output) override {
      size_t input_size = input.size();
      for (size_t i = 0; i < input_size; ++i) {
        output[i] = (input[i] > 0) ? input[i] : alpha_ * input[i];
      }
    }

    void BackwardActivate(const std::vector<dataType> &input, const std::vector<dataType> &output,
                          const std::vector<dataType> &delta, std::vector<dataType> &prev_delta) override {
      size_t input_size = input.size();
      for (size_t i = 0; i < input_size; ++i) {
        prev_delta[i] = (output[i] > 0) ? delta[i] : alpha_ * delta[i];
      }
    }

    void Print(std::ostream &os, bool) const override {
      os << "ID: " << this->layer_id_ << " LeakyReLU activation: " << this->input_size_ << " -> "
         << this->output_size_ << std::endl;
    }

    void Save(std::ofstream &file) const override {
      file.write(reinterpret_cast<const char *>(&LayerTypeTraits<LeakyReLuActivation<dataType>>::type),
                 sizeof(LayerTypeTraits<LeakyReLuActivation<dataType>>::type));
      file.write(reinterpret_cast<const char *>(&this->input_size_), sizeof(this->input_size_));
      file.write(reinterpret_cast<const char *>(&alpha_), sizeof(alpha_));
    }

  };

}