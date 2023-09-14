#pragma once

#include <NeuralNet/misc/math_util.h>
#include <NeuralNet/Layers/base_trainable_layer.h>

namespace NeuralNet {

  template<std::floating_point dataType = NNFLOAT>
  class FullyConnectedLayer;

  template<std::floating_point T>
  struct LayerTypeTraits<FullyConnectedLayer<T>> {
    static constexpr LayerType type = LayerType::FullyConnected;
  };

  template<std::floating_point dataType>
  class FullyConnectedLayer : public BaseTrainableLayer<dataType> {
   private:

    std::vector<dataType> weights_biases_;

    const dataType *weights_;
    const dataType *biases_;

   public:

    FullyConnectedLayer(size_t input_size, size_t output_size) :
        BaseTrainableLayer<dataType>(input_size, output_size), weights_biases_(input_size * output_size + output_size) {
      weights_ = weights_biases_.data();
      biases_ = weights_ + this->input_size_ * this->output_size_;
    }

    [[nodiscard]] size_t ParametersSize() const override {
      return weights_biases_.size();
    }

    [[nodiscard]] std::vector<dataType> &Parameters() override {
      return weights_biases_;
    }

    void Forward(const std::vector<std::vector<dataType> *> &inputs,
                 std::vector<std::vector<dataType> *> &outputs) override {
      this->ForwardAssert(inputs, outputs);

      const size_t inputs_size = inputs.size();
      const size_t input_size = this->input_size_;
      const size_t output_size = this->output_size_;

      for (int i = 0; i < inputs_size; ++i) {
        const dataType *input = inputs[i]->data();
        std::vector<dataType> &output = *(outputs[i]);

        for (size_t j = 0; j < output_size; ++j) {
          output[j] = MathUtil::Dot(input, weights_ + j * input_size, input_size) + biases_[j];
        }
      }
    }

    void Backward(const std::vector<std::vector<dataType> *> &inputs,
                  const std::vector<std::vector<dataType> *> &outputs,
                  const std::vector<std::vector<dataType>> &deltas,
                  std::vector<std::vector<dataType>> &prev_deltas,
                  std::vector<dataType> &grad_weights) override {
      this->BackwardAssert(inputs, outputs, deltas, prev_deltas, grad_weights);

      const size_t inputs_size = inputs.size();
      const size_t input_size = this->input_size_;
      const size_t output_size = this->output_size_;

      dataType *grad_weights_data = grad_weights.data();
      dataType *grad_biases_data = grad_weights_data + input_size * output_size;

      for (size_t i = 0; i < inputs_size; ++i) {
        std::vector<dataType> &prev_delta = prev_deltas[i];
        const std::vector<dataType> &delta = deltas[i];

        for (size_t j = 0; j < input_size; ++j) {
          dataType sum = 0;

          const dataType *weights_data_shifted = weights_ + j;

          for (size_t k = 0; k < output_size; ++k) {
            sum += delta[k] * weights_data_shifted[k * input_size];
          }

          prev_delta[j] = sum;
        }
      }

      for (size_t i = 0; i < output_size; ++i) {
        for (size_t j = 0; j < inputs_size; ++j) {
          const std::vector<dataType> &input = *(inputs[j]);
          dataType *grad_weights_data_shifted = grad_weights_data + i * input_size;
          const dataType delta_value = deltas[j][i];

          for (size_t k = 0; k < input_size; ++k) {
            grad_weights_data_shifted[k] += delta_value * input[k];
          }
        }
      }

      for (size_t i = 0; i < inputs_size; ++i) {
        const std::vector<dataType> &delta = deltas[i];

        for (size_t j = 0; j < output_size; ++j) {
          grad_biases_data[j] += delta[j];
        }
      }
    }

    void UpdateParameters(const std::vector<dataType> &updates) override {
      const size_t weights_biases_size = weights_biases_.size();
      for (size_t i = 0; i < weights_biases_size; ++i) {
        weights_biases_[i] += updates[i];
      }
    }

    void Print(std::ostream &os, bool weights) const override {
      const size_t input_size = this->input_size_;
      const size_t output_size = this->output_size_;

      os << "ID: " << this->layer_id_ << " FullyConnectedLayer: " << input_size << " -> "
         << output_size << std::endl;

      if (weights) {
        os << "Parameters: " << std::endl;
        for (size_t i = 0; i < output_size; ++i) {
          for (size_t j = 0; j < input_size; ++j) {
            os << weights_[i * input_size + j] << " ";
          }
          os << std::endl;
        }
        os << "Biases: " << std::endl;
        for (size_t i = 0; i < output_size; ++i) {
          os << biases_[i] << " ";
        }
        os << std::endl;
      }
    }

    void Save(std::ofstream &file) const override {
      file.write(reinterpret_cast<const char *>(&LayerTypeTraits<FullyConnectedLayer<dataType>>::type),
                 sizeof(LayerTypeTraits<FullyConnectedLayer<dataType>>::type));
      file.write(reinterpret_cast<const char *>(&this->input_size_), sizeof(this->input_size_));
      file.write(reinterpret_cast<const char *>(&this->output_size_), sizeof(this->output_size_));
      file.write(reinterpret_cast<const char *>(weights_biases_.data()),
                 sizeof(dataType) * weights_biases_.size());
    }
  };

}