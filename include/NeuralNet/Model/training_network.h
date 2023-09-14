#pragma once

#include <memory>
#include <vector>
#include <algorithm>
#include <functional>

#include <NeuralNet/Model/inference_network.h>
#include <NeuralNet/Optimizers/base_optimizer.h>
#include <NeuralNet/CostFunctions/base_cost.h>
#include <NeuralNet/Initializers/base_initializer.h>

namespace NeuralNet::Training {

  template<std::floating_point dataType = NNFLOAT>
  class NetworkTraining : public NetworkInference<dataType> {
   private:
    std::shared_ptr<BaseOptimizer<dataType>> optimizer_;
    std::shared_ptr<BaseCost<dataType>> cost_function_;
    std::vector<std::shared_ptr<BaseInitializer<dataType>>> initializers_;
    std::vector<std::shared_ptr<BaseLogger<dataType>>> loggers_;

   private:
    std::vector<std::vector<std::vector<dataType>>> train_outputs_storage_;
    std::vector<std::vector<std::vector<dataType> *>> train_outputs_;

    std::vector<std::vector<std::vector<dataType>>> deltas_;

    std::vector<std::vector<dataType>> input_deltas_;

    std::vector<std::vector<dataType> *> train_inputs_;
    std::vector<std::vector<dataType> *> target_outputs_;

    std::vector<std::vector<dataType>> grad_weights_;

    std::vector<std::vector<NNFLOAT>> test_inputs_;
    std::vector<std::vector<NNFLOAT>> test_target_outputs;

    long long training_iterations_ = 0;

    dataType last_train_cost_ = 0;
    dataType last_test_cost_ = 0;
    long long last_train_cost_computed_at = std::numeric_limits<long long>::min();
    long long last_test_cost_computed_at = std::numeric_limits<long long>::min();

    std::mt19937 &gen_;

   public:

    NetworkTraining(const std::shared_ptr<NeuralNetwork<dataType>> &network,
                    const std::shared_ptr<BaseOptimizer<dataType>> &optimizer,
                    const std::shared_ptr<BaseCost<dataType>> &cost_function,
                    const std::vector<std::shared_ptr<BaseInitializer<dataType>>> &initializers,
                    std::mt19937 &gen,
                    const std::vector<std::shared_ptr<BaseLogger<dataType>>> &loggers = {},
                    bool initialize_weights = true) :
        NetworkInference<dataType>(network),
        optimizer_(optimizer),
        cost_function_(cost_function),
        initializers_(initializers),
        gen_(gen),
        loggers_(loggers) {

      ConstructorHelper(initialize_weights);
    }

    [[nodiscard]] long long TrainingIterations() const {
      return training_iterations_;
    }

    [[nodiscard]] dataType LastTrainCost() const {
      return last_train_cost_;
    }

    [[nodiscard]] dataType LastTestCost() const {
      return last_test_cost_;
    }

    [[nodiscard]] long long LastTrainCostComputedAtIteration() const {
      return last_train_cost_computed_at;
    }

    [[nodiscard]] long long LastTestCostComputedAtIteration() const {
      return last_test_cost_computed_at;
    }

    [[nodiscard]] std::vector<std::shared_ptr<BaseLogger<dataType>>> &Loggers() {
      return loggers_;
    }

    void ConstructorHelper(bool initialize_weights) {
      if (initialize_weights) {
        InitializeParameters();
      }

      optimizer_->Allocate(this->network_);

      for (const auto &layer : *this->network_) {
        grad_weights_.emplace_back(layer->ParametersSize());
      }
    }

    void SetTest(const std::vector<std::vector<NNFLOAT>> &inputs,
                 const std::vector<std::vector<NNFLOAT>> &outputs) {
      assert(inputs.size() == outputs.size());
      test_inputs_ = inputs;
      test_target_outputs = outputs;
    }

    void InitializeParameters() {
      training_iterations_ = 0;
      last_train_cost_computed_at = std::numeric_limits<long long>::min();
      last_test_cost_computed_at = std::numeric_limits<long long>::min();

      for (auto initializer : initializers_) {
        initializer->InitializeTrainableParams(this->network_, gen_);
      }
    }

    void TrainSample(const std::vector<dataType> &input, const std::vector<dataType> &target_output) {
      assert(this->network_->LayersSize() != 0);
      assert(input.size() == this->network_->InputSize());
      assert(target_output.size() == this->network_->OutputSize());

      AllocateTrainVectors(1);

      train_inputs_[0] = const_cast<std::vector<dataType> *>( &input );
      target_outputs_[0] = const_cast<std::vector<dataType> *>( &target_output );

      RunTraining();

      for (const auto &logger_ : loggers_) {
        logger_->Log(this, training_iterations_);
      }
      if (!loggers_.empty()) {
        loggers_[0]->IncreaseIteration();
      }
      ++training_iterations_;
    }

    void TrainBatch(const std::vector<std::vector<dataType>> &inputs,
                    const std::vector<std::vector<dataType>> &target_outputs) {
      assert(this->network_->LayersSize() != 0);
      assert(!inputs.empty());
      assert(inputs.size() == target_outputs.size());
      for (const auto &input : inputs) {
        assert(input.size() == this->network_->InputSize());
      }
      for (const auto &output : target_outputs) {
        assert(output.size() == this->network_->OutputSize());
      }

      const size_t inputs_size = inputs.size();

      AllocateTrainVectors(inputs_size);

      for (size_t i = 0; i < inputs_size; ++i) {
        train_inputs_[i] = const_cast<std::vector<dataType> *>( &(inputs[i]));
        target_outputs_[i] = const_cast<std::vector<dataType> *>( &(target_outputs[i]));
      }

      RunTraining();

      for (const auto &logger_ : loggers_) {
        logger_->Log(this);
      }
      training_iterations_ += inputs_size;
    }

    void TrainEpoch(const std::vector<std::vector<dataType>> &inputs,
                    const std::vector<std::vector<dataType>> &target_outputs,
                    size_t batchSize, bool random) {
      assert(this->network_->LayersSize() != 0);
      assert(!inputs.empty());
      assert(inputs.size() == target_outputs.size());
      for (const auto &input : inputs) {
        assert(input.size() == this->network_->InputSize());
      }
      for (const auto &output : target_outputs) {
        assert(output.size() == this->network_->OutputSize());
      }

      dataType average_cost = 0;
      const size_t inputs_size = inputs.size();

      AllocateTrainVectors(batchSize);

      if (inputs_size == batchSize) {
        TrainBatch(inputs, target_outputs);
      } else {
        const size_t iterations = (inputs_size - 1) / batchSize + 1;

        for (size_t i = 0; i < iterations; ++i) {
          for (size_t j = 0; j < batchSize; ++j) {
            size_t index;
            if (random) {
              index = gen_() % inputs_size;
            } else {
              index = (i * batchSize + j) % inputs_size;
            }

            train_inputs_[j] = const_cast<std::vector<dataType> *>( &(inputs[index]));
            target_outputs_[j] = const_cast<std::vector<dataType> *>( &(target_outputs[index]));
          }

          RunTraining();
        }

        for (const auto &logger_ : loggers_) {
          logger_->Log(this);
        }
        training_iterations_ += batchSize * iterations;
      }
    }

    dataType CalculateTrainCost() {
      const std::vector<std::vector<dataType>> &last_outputs = train_outputs_storage_.back();
      dataType total_cost = 0;
      const size_t train_inputs_size = train_inputs_.size();

      for (size_t i = 0; i < train_inputs_size; ++i) {
        const std::vector<dataType> &last_output = last_outputs[i];
        const std::vector<dataType> &target_output = *target_outputs_[i];

        total_cost += cost_function_->Cost(last_output, target_output);
      }

      total_cost /= static_cast<dataType>(train_inputs_size);

      last_train_cost_computed_at = training_iterations_;
      last_train_cost_ = total_cost;
      return total_cost;
    }

    dataType CalculateTestCost() {
      if (test_inputs_.empty()) {
        last_test_cost_computed_at = training_iterations_;
        last_test_cost_ = 0;
        return 0;
      }

      std::vector<dataType> test_output(this->network_->OutputSize());
      dataType total_cost = 0;
      const size_t test_inputs_size = test_inputs_.size();

      for (size_t i = 0; i < test_inputs_size; i++) {
        test_output = (*this)(test_inputs_[i]);

        total_cost += cost_function_->Cost(test_output, test_target_outputs[i]);
      }

      total_cost /= static_cast<dataType>(test_inputs_size);

      last_test_cost_computed_at = training_iterations_;
      last_test_cost_ = total_cost;
      return total_cost;
    }

   private:

    void RunTraining() {
      this->Forward(train_inputs_, train_outputs_);

      Backward();

      //average gradients
      const size_t layers_size = this->network_->LayersSize();
      for (size_t i = 0; i < layers_size; ++i) {
        for (auto &param : grad_weights_[i]) {
          param = param / train_inputs_.size();
        }
      }

      UpdateParams();
    }

    void Backward() {
      assert(this->network_->LayersSize() != 0);

      size_t train_inputs_size = train_inputs_.size();

      std::vector<std::vector<dataType>> &last_outputs = train_outputs_storage_.back();
      std::vector<std::vector<dataType>> &last_deltas = deltas_.back();

      for (size_t i = 0; i < train_inputs_size; ++i) {
        std::vector<dataType> &last_delta = last_deltas[i];
        std::vector<dataType> &last_output = last_outputs[i];
        std::vector<dataType> &target_output = *target_outputs_[i];

        last_delta = cost_function_->Gradient(last_output, target_output);
      }

      size_t layer_index = this->network_->LayersSize() - 1;

      while (layer_index > 0) {
        this->network_->LayerAt(layer_index)->
            Backward(train_outputs_[layer_index - 1],
                     train_outputs_[layer_index],
                     deltas_[layer_index],
                     deltas_[layer_index - 1],
                     grad_weights_[layer_index]);
        --layer_index;
      }

      this->network_->LayerAt(0)->
          Backward(train_inputs_,
                   train_outputs_[0],
                   deltas_[0],
                   input_deltas_,
                   grad_weights_[0]);
    }

    void UpdateParams() {
      const size_t layers_size = this->network_->LayersSize();

      for (size_t i = 0; i < layers_size; ++i) {
        auto layer = this->network_->LayerAt(i);
        if (layer->Trainable()) {
          optimizer_->CalculateUpdatesFromGradients(grad_weights_[i], i); //i == layer_ID

          std::static_pointer_cast<BaseTrainableLayer<dataType>>(layer)->UpdateParameters(grad_weights_[i]);

          for (auto &param : grad_weights_[i]) {
            param = 0;
          }
        }
      }
    }

    void AllocateTrainVectors(size_t samples_count) {
      const size_t layers_size = this->network_->LayersSize();

      if (train_inputs_.size() != samples_count) {
        train_inputs_.resize(samples_count);
        target_outputs_.resize(samples_count);

        train_outputs_storage_.resize(layers_size);
        train_outputs_.resize(layers_size);

        deltas_.resize(layers_size);

        for (size_t layer_index = 0; layer_index < layers_size; ++layer_index) {
          size_t layer_output_count = this->network_->LayerAt(layer_index)->OutputSize();

          train_outputs_storage_[layer_index].resize(samples_count);
          train_outputs_[layer_index].resize(samples_count);

          deltas_[layer_index].resize(samples_count);

          for (size_t i = 0; i < samples_count; ++i) {
            train_outputs_storage_[layer_index][i] = std::vector<dataType>(layer_output_count);
            train_outputs_[layer_index][i] = &(train_outputs_storage_[layer_index][i]);

            deltas_[layer_index][i] = std::vector<dataType>(layer_output_count);
          }
        }

        input_deltas_.resize(samples_count);

        for (size_t i = 0; i < samples_count; ++i) {
          input_deltas_[i] = std::vector<dataType>(this->network_->InputSize());
        }
      }
    }
  };

}