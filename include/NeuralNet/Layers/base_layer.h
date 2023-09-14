#pragma once

#include <iostream>
#include <cstdio>
#include <fstream>
#include <cassert>

#include <NeuralNet/misc/types.h>
#include <NeuralNet/misc/layer_type.h>

namespace NeuralNet {

  template<std::floating_point dataType = NNFLOAT>
  class NeuralNetwork;

  template<std::floating_point dataType>
  class BaseLayer {
   protected:

    size_t input_size_;
    size_t output_size_;

    size_t layer_id_ = 0;

   public:

    BaseLayer(size_t input_size, size_t output_size) : input_size_(input_size), output_size_(output_size) {}

    virtual ~BaseLayer() = default;

    [[nodiscard]] size_t InputSize() const {
      return input_size_;
    }

    [[nodiscard]] size_t OutputSize() const {
      return output_size_;
    }

    [[nodiscard]] size_t LayerID() const {
      return layer_id_;
    }

    [[nodiscard]] virtual size_t ParametersSize() const = 0;

    [[nodiscard]] virtual bool Trainable() const = 0;

    virtual void Forward(const std::vector<std::vector<dataType> *> &inputs,
                         std::vector<std::vector<dataType> *> &outputs) = 0;

    virtual void Backward(const std::vector<std::vector<dataType> *> &inputs,
                          const std::vector<std::vector<dataType> *> &outputs,
                          const std::vector<std::vector<dataType>> &deltas,
                          std::vector<std::vector<dataType>> &prev_deltas,
                          std::vector<dataType> &grad_weights) = 0;

    virtual void Print(std::ostream &os, bool weights) const = 0;

    virtual void Save(std::ofstream &os) const = 0;

    friend class NeuralNetwork<dataType>;

   protected:

    void ForwardAssert(const std::vector<std::vector<dataType> *> &inputs,
                       const std::vector<std::vector<dataType> *> &outputs) const {
      assert(inputs.size() == outputs.size());
      for (int i = 0; i < inputs.size(); ++i) {
        assert(inputs[i]->size() == this->input_size_);
      }
      for (int i = 0; i < outputs.size(); ++i) {
        assert(outputs[i]->size() == this->output_size_);
      }
    }

    void BackwardAssert(const std::vector<std::vector<dataType> *> &inputs,
                        const std::vector<std::vector<dataType> *> &outputs,
                        const std::vector<std::vector<dataType>> &deltas,
                        const std::vector<std::vector<dataType>> &prev_deltas,
                        const std::vector<dataType> &) const {
      assert(inputs.size() == outputs.size());
      assert(inputs.size() == deltas.size());
      assert(inputs.size() == prev_deltas.size());
      for (int i = 0; i < inputs.size(); ++i) {
        assert(inputs[i]->size() == this->input_size_);
      }
      for (int i = 0; i < outputs.size(); ++i) {
        assert(outputs[i]->size() == this->output_size_);
      }
      for (int i = 0; i < deltas.size(); ++i) {
        assert(deltas[i].size() == this->output_size_);
      }
      for (int i = 0; i < prev_deltas.size(); ++i) {
        assert(prev_deltas[i].size() == this->input_size_);
      }
    }
  };

}