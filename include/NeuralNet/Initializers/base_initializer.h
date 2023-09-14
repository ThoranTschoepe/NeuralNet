#pragma once

#include <NeuralNet/Model/base_network.h>

#include <utility>

namespace NeuralNet::Training {

  template<std::floating_point dataType>
  class BaseInitializer {
   protected:

    std::vector<size_t> ids_;

   public:

    explicit BaseInitializer(const std::vector<size_t> &ids = {}) : ids_(ids) {}

    void InitializeTrainableParams(const std::shared_ptr<NeuralNetwork<dataType>> &network, std::mt19937 &gen) {
      for (auto &layer : *network) {
        //check if we need to initialize this layer
        if (ids_.empty() || std::find(ids_.begin(), ids_.end(), layer->LayerID()) != ids_.end()) {
          InitializeLayer(layer, gen);
        }
      }
    }

   protected:

    virtual void InitializeLayer(std::shared_ptr<BaseLayer<dataType>> &layer, std::mt19937 &gen) = 0;
  };

}