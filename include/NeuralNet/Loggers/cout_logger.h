#pragma once

#include <NeuralNet/Loggers/baser_logger.h>

namespace NeuralNet::Training {

  template<std::floating_point dataType>
  class coutLogger : public BaseLogger<dataType> {
    size_t mod_;
    long long last_logged_iteration_ = std::numeric_limits<long long>::min();

   public:

    explicit coutLogger(size_t mod = 99999) : mod_(mod) {}

    void Log(NetworkTraining<dataType> *network_training) override {
      long long current_iteration = network_training->TrainingIterations();
      if (current_iteration - last_logged_iteration_ > mod_) {
        this->Compute(network_training);
        last_logged_iteration_ = current_iteration;

        std::cout << "Iteration: " << current_iteration << " Train cost: " << network_training->LastTrainCost()
                  << " Test cost: " << network_training->LastTestCost() << std::endl;
      }
    }

  };

}