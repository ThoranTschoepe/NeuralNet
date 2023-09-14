#pragma once

#include <memory>

#include <NeuralNet/misc/types.h>
#include <NeuralNet/misc/layer_type.h>
#include <NeuralNet/Layers/base_layer.h>
#include <NeuralNet/Layers/base_trainable_layer.h>

namespace NeuralNet {

  template<std::floating_point dataType>
  class NeuralNetwork {
   private:
    size_t layer_id_counter_ = 0;
    std::vector<std::shared_ptr<BaseLayer<dataType>>> layers_;

   public:

    typedef std::vector<std::shared_ptr<BaseLayer<dataType>>>::iterator iterator;
    typedef std::vector<std::shared_ptr<BaseLayer<dataType>>>::reverse_iterator reverse_iterator;
    typedef std::vector<std::shared_ptr<BaseLayer<dataType>>>::const_iterator const_iterator;

    NeuralNetwork() = default;

    [[nodiscard]] size_t InputSize() const {
      return (layers_.empty()) ? 0 : layers_.front()->InputSize();
    }

    [[nodiscard]] size_t OutputSize() const {
      return (layers_.empty()) ? 0 : layers_.back()->OutputSize();
    }

    [[nodiscard]] size_t LayersSize() const {
      return layers_.size();
    }

    [[nodiscard]] iterator begin() {
      return layers_.begin();
    }

    [[nodiscard]] iterator end() {
      return layers_.end();
    }

    [[nodiscard]] reverse_iterator rbegin() {
      return layers_.rbegin();
    }

    [[nodiscard]] reverse_iterator rend() {
      return layers_.rend();
    }

    [[nodiscard]] const_iterator begin() const {
      return layers_.begin();
    }

    [[nodiscard]] const_iterator end() const {
      return layers_.end();
    }

    [[nodiscard]] std::shared_ptr<BaseLayer<dataType>> LayerAt(size_t index) const {
      return (index < layers_.size()) ? layers_[index] : std::shared_ptr<BaseLayer<dataType>>(nullptr);
    }

    template<template<typename> typename Layer, typename... T>
    void AddLayer(T &&... args) {
      layers_.push_back(std::make_shared<Layer<dataType>>(std::forward<T>(args)...));
      layers_.back()->layer_id_ = layer_id_counter_++;
    }

    void AddLayer(std::shared_ptr<BaseLayer<dataType>> layer) {
      layers_.push_back(layer);
      layers_.back()->layer_id_ = layer_id_counter_++;
    }

    void Print(std::ostream &os = std::cout, bool weights = false) const {
      for (auto &layer : layers_) {
        layer->Print(os, weights);
      }
    }

    void Save(const std::string &filepath) const {
      std::ofstream os(filepath, std::ios::binary);
      if (!os.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
      }

      for (auto &layer : layers_) {
        layer->Save(os);
      }

      os.close();
    }
  };

  template<std::floating_point dataType = NNFLOAT>
  std::shared_ptr<NeuralNetwork<dataType>> Load(const std::string &filepath) {
    std::ifstream is(filepath, std::ios::binary);
    if (!is.is_open()) {
      throw std::runtime_error("Failed to open file for reading: " + filepath);
    }

    auto network = std::make_shared<NeuralNetwork<dataType>>();

    while (is.peek() != EOF) {
      int layerType;
      is.read(reinterpret_cast<char *>(&layerType), sizeof(layerType));

      switch (layerType) {
        case LayerType::FullyConnected: {
          size_t inputsCount, outputsCount;
          is.read(reinterpret_cast<char *>(&inputsCount), sizeof(inputsCount));
          is.read(reinterpret_cast<char *>(&outputsCount), sizeof(outputsCount));

          auto layer = std::make_shared<FullyConnectedLayer<dataType>>(inputsCount, outputsCount);

          layer->Parameters().resize(inputsCount * outputsCount + outputsCount);
          is.read(reinterpret_cast<char *>(layer->Parameters().data()), sizeof(dataType) * layer->Parameters().size());

          network->AddLayer(layer);
          break;
        }
        case LayerType::LeakyReLU : {
          size_t inputsCount;
          is.read(reinterpret_cast<char *>(&inputsCount), sizeof(inputsCount));

          auto layer = std::make_shared<LeakyReLuActivation<dataType>>(inputsCount, inputsCount);
          network->AddLayer(layer);
          break;
        }
        case LayerType::ReLU : {
          size_t inputsCount;
          is.read(reinterpret_cast<char *>(&inputsCount), sizeof(inputsCount));

          auto layer = std::make_shared<ReLuActivation<dataType>>(inputsCount, inputsCount);
          network->AddLayer(layer);
          break;
        }
        case LayerType::Sigmoid: {
          size_t inputsCount;
          is.read(reinterpret_cast<char *>(&inputsCount), sizeof(inputsCount));

          auto layer = std::make_shared<SigmoidActivation<dataType>>(inputsCount, inputsCount);
          network->AddLayer(layer);
          break;
        }
        case LayerType::Softmax: {
          size_t inputsCount;
          is.read(reinterpret_cast<char *>(&inputsCount), sizeof(inputsCount));

          auto layer = std::make_shared<SoftmaxActivation<dataType>>(inputsCount, inputsCount);
          network->AddLayer(layer);
          break;
        }
        case LayerType::Tanh: {
          size_t inputsCount;
          is.read(reinterpret_cast<char *>(&inputsCount), sizeof(inputsCount));

          auto layer = std::make_shared<TanhActivation<dataType>>(inputsCount, inputsCount);
          network->AddLayer(layer);
          break;
        }
        default:break;
      }
    }

    return network;
  }

}