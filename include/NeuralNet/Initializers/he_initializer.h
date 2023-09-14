#pragma once

#include <NeuralNet/Initializers/base_initializer.h>

namespace NeuralNet::Training {

  template<std::floating_point dataType = NNFLOAT>
  class HeInitializer : public BaseInitializer<dataType> {
   public:

    explicit HeInitializer(const std::vector<size_t> &ids = {}) : BaseInitializer<dataType>(ids) {}

   private:

    void InitializeLayer(std::shared_ptr<BaseLayer<dataType>> &layer, std::mt19937 &gen) override {
      if (auto fcl = dynamic_cast<FullyConnectedLayer<dataType> *>(layer.get())) {
        const size_t input_size = fcl->InputSize();
        const size_t output_size = fcl->OutputSize();
        const size_t sep = input_size * output_size;

        std::vector<dataType> &parameters = fcl->Parameters();

        std::normal_distribution<dataType> dist(0, sqrt(2 / static_cast<dataType>(input_size)));

        for (size_t i = 0; i < sep; ++i) {
          parameters[i] = dist(gen);
        }
        for (size_t i = sep; i < sep + output_size; ++i) {
          parameters[i] = 0;
        }
      }
    }
  };

}