#pragma once

#include <algorithm>

#include <NeuralNet/Layers/Activations/leaky_relu_activation.h>
#include <NeuralNet/Layers/Activations/relu_activation.h>
#include <NeuralNet/Layers/Activations/sigmoid_activation.h>
#include <NeuralNet/Layers/Activations/softmax_activation.h>
#include <NeuralNet/Layers/Activations/tanh_activation.h>

#include <NeuralNet/Layers/fully_connected_layer.h>

#include <NeuralNet/Model/base_network.h>
#include <NeuralNet/misc/layer_type.h>

namespace NeuralNet {

  template<std::floating_point dataType = NNFLOAT>
  class NetworkBuilder {
    std::vector<LayerType> layer_types_;
    std::vector<int> layer_outputs_;
    std::vector<std::string> activations_;

    std::shared_ptr<NeuralNetwork<dataType>> network_;

   public:

    explicit NetworkBuilder(int inputs) {
      layer_types_.push_back(LayerType::DEFAULT);
      layer_outputs_.push_back(inputs);
      activations_.emplace_back("");
      network_ = std::make_shared<NeuralNetwork<dataType>>();
    }

    template<template<typename> typename Layer>
    void AddLayer(int outputs, const std::string &activation = "") {
      layer_types_.push_back(LayerTypeTraits<Layer<NNFLOAT>>::type);
      layer_outputs_.push_back(outputs);
      activations_.push_back(activation);
      toLower(activations_.back());
    }

    std::shared_ptr<NeuralNetwork<dataType>> Build() {
      for (int i = 1; i < layer_types_.size(); ++i) {
        switch (layer_types_[i]) {
          case LayerType::FullyConnected:
            network_->template AddLayer<FullyConnectedLayer>(layer_outputs_[i - 1],
                                                             layer_outputs_[i]);
            addActivation(activations_[i], layer_outputs_[i]);
            break;
          default:std::cout << "Layer type not recognized" << std::endl;
            break;
        }
      }

      return network_;
    }

   private:

    void toLower(std::string &str) {
      std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::tolower(c); });
    }

    void addActivation(const std::string &activation, int inputs) {
      if (activation == "leakyrelu") {
        network_->template AddLayer<LeakyReLuActivation>(inputs, inputs);
      } else if (activation == "relu") {
        network_->template AddLayer<ReLuActivation>(inputs, inputs);
      } else if (activation == "sigmoid") {
        network_->template AddLayer<SigmoidActivation>(inputs, inputs);
      } else if (activation == "softmax") {
        network_->template AddLayer<SoftmaxActivation>(inputs, inputs);
      } else if (activation == "tanh") {
        network_->template AddLayer<TanhActivation>(inputs, inputs);
      } else if (!activation.empty()) {
        std::cout << "Activation not recognized" << std::endl;
      }
    }

  };

}

