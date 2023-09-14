#pragma once

#include <iostream>

namespace NeuralNet::Training {

  template<std::floating_point dataType>
  class NetworkTraining;

  template<std::floating_point dataType>
  class BaseLogger {
   public:

    virtual void Log(NetworkTraining<dataType> *network_training) = 0;

   protected:

    void Compute(NetworkTraining<dataType> *network_training) {
      size_t current_iteration = network_training->TrainingIterations();

      size_t last_train_cost_computed_at_iteration = network_training->LastTrainCostComputedAtIteration();
      size_t last_test_cost_computed_at_iteration = network_training->LastTestCostComputedAtIteration();

      if (last_train_cost_computed_at_iteration != current_iteration) {
        network_training->CalculateTrainCost();
      }
      if (last_test_cost_computed_at_iteration != current_iteration) {
        network_training->CalculateTestCost();
      }
    }

  };

}