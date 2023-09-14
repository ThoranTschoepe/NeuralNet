#pragma once

#include <NeuralNet/Loggers/baser_logger.h>

#include <fstream>
#include <utility>
#include <iomanip>

namespace NeuralNet::Training {

  template<std::floating_point dataType>
  class CSVLogger : public BaseLogger<dataType> {
    size_t mod_;
    long long last_logged_iteration_ = std::numeric_limits<long long>::min();

    std::string filename_;
    std::ofstream file_;
   public:

    explicit CSVLogger(std::string filename, size_t mod = 99999) : filename_(std::move(filename)), mod_(mod) {
      file_.open(filename_);

      file_ << "Iteration,TrainCost,TestCost" << std::endl;
    }

    void Log(NetworkTraining<dataType> *network_training) override {
      size_t current_iteration = network_training->TrainingIterations();
      if (current_iteration - last_logged_iteration_ > mod_) {
        this->Compute(network_training);
        last_logged_iteration_ = current_iteration;

        file_ << std::fixed << std::setprecision(std::numeric_limits<dataType>::max_digits10) << current_iteration << "," << network_training->LastTrainCost() << "," << network_training->LastTestCost() << std::endl;
      }
    }

    void SetFile(std::string filename) {
      if(filename == filename_) return;
      if(file_.is_open()) file_.close();
      filename_ = std::move(filename);
      file_.open(filename_);
    }

    void WriteHead(){
      file_ << "Iteration,TrainCost,TestCost" << std::endl;
    }

    ~CSVLogger() {
      file_.close();
    }

  };

}