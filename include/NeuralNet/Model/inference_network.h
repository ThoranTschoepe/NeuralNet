#pragma once

#include <memory>
#include <vector>
#include <cassert>

#include <NeuralNet/Model/base_network.h>

namespace NeuralNet {

  template<std::floating_point dataType = NNFLOAT>
  class NetworkInference {
   protected:
    std::shared_ptr<NeuralNetwork<dataType>> network_;

   protected:
    std::vector<std::vector<std::vector<dataType>>> compute_outputs_storage_;
    std::vector<std::vector<std::vector<dataType> *>> compute_outputs_;
    std::vector<std::vector<dataType> *> compute_inputs_;

   public:
    explicit NetworkInference(const std::shared_ptr<NeuralNetwork<dataType>> &network) : network_(network) {
      compute_inputs_.resize(1);

      for (const auto &layer : *network_) {
        compute_outputs_storage_.push_back(
            std::vector<std::vector<dataType>>({std::vector<dataType>(layer->OutputSize())}));
        compute_outputs_.push_back(std::vector<std::vector<dataType> *>({&(compute_outputs_storage_.back()[0])}));
      }

    }

    std::vector<dataType> &operator()(const std::vector<dataType> &input) {
      compute_inputs_[0] = const_cast<std::vector<dataType> *>( &input );

      Forward(compute_inputs_, compute_outputs_);

      return compute_outputs_storage_.back()[0];
    }

   protected:

    void Forward(const std::vector<std::vector<dataType> *> &inputs,
                 std::vector<std::vector<std::vector<dataType> *>> &outputs) {
      assert(this->network_->LayersSize() != 0);

      size_t layers_size = network_->LayersSize();

      network_->LayerAt(0)->Forward(inputs, outputs[0]);

      for (size_t i = 1; i < layers_size; ++i) {
        network_->LayerAt(i)->Forward(outputs[i - 1], outputs[i]);
      }
    }

  };

}