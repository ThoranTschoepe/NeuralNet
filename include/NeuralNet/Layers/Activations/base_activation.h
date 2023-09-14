#pragma once

#include <NeuralNet/Layers/base_layer.h>

namespace NeuralNet {

  template<std::floating_point dataType>
  class BaseActivation : public BaseLayer<dataType> {
   public:

    BaseActivation(size_t input_size, size_t output_size) : BaseLayer<dataType>(input_size, output_size) {}

    [[nodiscard]] bool Trainable() const override {
      return false;
    }

    [[nodiscard]] size_t ParametersSize() const override {
      return 0;
    }

    void Forward(const std::vector<std::vector<dataType> *> &inputs,
                 std::vector<std::vector<dataType> *> &outputs) override {
      this->ForwardAssert(inputs, outputs);

      const size_t inputs_size = inputs.size();
      for (int i = 0; i < inputs_size; ++i) {
        ForwardActivate(*(inputs[i]), *(outputs[i]));
      }
    }

    void Backward(const std::vector<std::vector<dataType> *> &inputs,
                  const std::vector<std::vector<dataType> *> &outputs,
                  const std::vector<std::vector<dataType>> &deltas,
                  std::vector<std::vector<dataType>> &prev_deltas,
                  std::vector<dataType> &grad_weights) override {
      this->BackwardAssert(inputs, outputs, deltas, prev_deltas, grad_weights);

      const size_t inputs_size = inputs.size();
      for (int i = 0; i < inputs_size; ++i) {
        BackwardActivate(*(inputs[i]), *(outputs[i]), deltas[i], prev_deltas[i]);
      }
    }

    virtual void ForwardActivate(const std::vector<dataType> &input, std::vector<dataType> &output) = 0;

    virtual void BackwardActivate(const std::vector<dataType> &input, const std::vector<dataType> &output,
                                  const std::vector<dataType> &delta, std::vector<dataType> &prev_delta) = 0;
  };

}