#include <NeuralNet/NeuralNet.h>

#include <filesystem>

using namespace NeuralNet;
using namespace NeuralNet::Training;

std::pair<std::vector<std::vector<NNFLOAT>>, std::vector<std::vector<NNFLOAT>>> GenerateSinData(size_t count) {
  std::vector<std::vector<NNFLOAT>> inputs;
  std::vector<std::vector<NNFLOAT>> target_outputs;

  inputs.reserve(count);
  target_outputs.reserve(count);

  for (size_t i = 0; i < count; ++i) {
    NNFLOAT x = static_cast<NNFLOAT>( i * 2 * M_PI) / static_cast<NNFLOAT>(count);
    inputs.push_back({x});
    target_outputs.push_back({std::sin(x)});
  }

  return {inputs, target_outputs};
}

std::shared_ptr<NeuralNetwork<NNFLOAT>> CreateNetwork() {
  auto netBuilder = NetworkBuilder<NNFLOAT>(1);

  netBuilder.AddLayer<FullyConnectedLayer>(12, "relu");
  netBuilder.AddLayer<FullyConnectedLayer>(12, "relu");
  netBuilder.AddLayer<FullyConnectedLayer>(1, "");

  auto net = netBuilder.Build();

  return net;
}

void CreateFileSystem(const std::string &loc) {
  //check if adam directory exists, delete if it does and create a new one
  if (std::filesystem::exists(loc + "/data")) {
    std::filesystem::remove_all(loc + "/data");
  }

  std::filesystem::create_directory(loc + "/data");
}

auto net = CreateNetwork();

std::mt19937 gen = std::mt19937(9);

void RmsProp(const std::vector<std::vector<NNFLOAT>>& inputs, const std::vector<std::vector<NNFLOAT>>& target_outputs, size_t n_networks_train, const std::string &loc) {
  NetworkTraining<NNFLOAT> rmsprop_training(net,
                                            std::make_shared<RMSPropOptimizer<NNFLOAT>>(),
                                            std::make_shared<MSECost<NNFLOAT>>(),
                                            std::vector<std::shared_ptr<BaseInitializer<NNFLOAT>>>{
                                                std::make_shared<NormalizedUniformXavierInitializer<NNFLOAT>>()},
                                            gen,
                                            std::vector<std::shared_ptr<BaseLogger<NNFLOAT>>>{
                                                std::make_shared<CSVLogger<NNFLOAT>>("temp.csv", 999)});

  std::filesystem::create_directory(loc + "/data/rmsprop");

  for (size_t i = 0; i < n_networks_train; ++i) {
    auto csv_logger = dynamic_cast<CSVLogger<NNFLOAT> *>(rmsprop_training.Loggers()[0].get());
    csv_logger->SetFile(loc + "/data/rmsprop/" + std::to_string(i) + ".csv");
    csv_logger->WriteHead();

    rmsprop_training.InitializeParameters();
    for (size_t j = 0; j < 1000; ++j)
      rmsprop_training.TrainEpoch(inputs, target_outputs, 128, true);
  }

  std::cout << "RMSProp finished" << std::endl;
}

void Adam(const std::vector<std::vector<NNFLOAT>>& inputs, const std::vector<std::vector<NNFLOAT>>& target_outputs, size_t n_networks_train, const std::string &loc) {
  NetworkTraining<NNFLOAT> adam_training(net,
                                         std::make_shared<AdamOptimizer<NNFLOAT>>(),
                                         std::make_shared<MSECost<NNFLOAT>>(),
                                         std::vector<std::shared_ptr<BaseInitializer<NNFLOAT>>>{
                                             std::make_shared<NormalizedUniformXavierInitializer<NNFLOAT>>()},
                                         gen,
                                         std::vector<std::shared_ptr<BaseLogger<NNFLOAT>>>{
                                             std::make_shared<CSVLogger<NNFLOAT>>("temp.csv", 999)});
  //create file directory
  std::filesystem::create_directory(loc + "/data/adam");

  for (size_t i = 0; i < n_networks_train; ++i) {
    auto csv_logger = dynamic_cast<CSVLogger<NNFLOAT> *>(adam_training.Loggers()[0].get());
    csv_logger->SetFile(loc + "/data/adam/" + std::to_string(i) + ".csv");
    csv_logger->WriteHead();

    adam_training.InitializeParameters();
    for (size_t j = 0; j < 1000; ++j)
      adam_training.TrainEpoch(inputs, target_outputs, 128, true);
  }
  std::cout << "Adam finished" << std::endl;
}

void SGD(const std::vector<std::vector<NNFLOAT>>& inputs, const std::vector<std::vector<NNFLOAT>>& target_outputs, size_t n_networks_train, const std::string &loc) {
  NetworkTraining<NNFLOAT> sgd_training(net,
                                        std::make_shared<GradOptimizer<NNFLOAT>>(),
                                        std::make_shared<MSECost<NNFLOAT>>(),
                                        std::vector<std::shared_ptr<BaseInitializer<NNFLOAT>>>{
                                            std::make_shared<NormalizedUniformXavierInitializer<NNFLOAT>>()},
                                        gen,
                                        std::vector<std::shared_ptr<BaseLogger<NNFLOAT>>>{
                                            std::make_shared<CSVLogger<NNFLOAT>>("temp.csv", 999)});

  std::filesystem::create_directory(loc + "/data/sgd");

  for (size_t i = 0; i < n_networks_train; ++i) {
    auto csv_logger = dynamic_cast<CSVLogger<NNFLOAT> *>(sgd_training.Loggers()[0].get());
    csv_logger->SetFile(loc + "/data/sgd/" + std::to_string(i) + ".csv");
    csv_logger->WriteHead();

    sgd_training.InitializeParameters();
    for (size_t j = 0; j < 1000; ++j)
      sgd_training.TrainEpoch(inputs, target_outputs, 128, true);
  }

  std::cout << "SGD finished" << std::endl;
}

void Momentum(const std::vector<std::vector<NNFLOAT>>& inputs, const std::vector<std::vector<NNFLOAT>>& target_outputs, size_t n_networks_train, const std::string &loc) {
  NetworkTraining<NNFLOAT> momentum_training(net,
                                             std::make_shared<MomentumOptimizer<NNFLOAT>>(),
                                             std::make_shared<MSECost<NNFLOAT>>(),
                                             std::vector<std::shared_ptr<BaseInitializer<NNFLOAT>>>{
                                                 std::make_shared<NormalizedUniformXavierInitializer<NNFLOAT>>()},
                                             gen,
                                             std::vector<std::shared_ptr<BaseLogger<NNFLOAT>>>{
                                                 std::make_shared<CSVLogger<NNFLOAT>>("temp.csv", 999)});

  std::filesystem::create_directory(loc + "/data/momentum");

  for (size_t i = 0; i < n_networks_train; ++i) {
    auto csv_logger = dynamic_cast<CSVLogger<NNFLOAT> *>(momentum_training.Loggers()[0].get());
    csv_logger->SetFile(loc + "/data/momentum/" + std::to_string(i) + ".csv");
    csv_logger->WriteHead();

    momentum_training.InitializeParameters();
    for (size_t j = 0; j < 1000; ++j)
      momentum_training.TrainEpoch(inputs, target_outputs, 128, true);
  }

  std::cout << "Momentum finished" << std::endl;
}

void Nesterov(const std::vector<std::vector<NNFLOAT>>& inputs, const std::vector<std::vector<NNFLOAT>>& target_outputs, size_t n_networks_train, const std::string &loc) {
  NetworkTraining<NNFLOAT> nesterov_training(net,
                                             std::make_shared<NesterovOptimizer<NNFLOAT>>(),
                                             std::make_shared<MSECost<NNFLOAT>>(),
                                             std::vector<std::shared_ptr<BaseInitializer<NNFLOAT>>>{
                                                 std::make_shared<NormalizedUniformXavierInitializer<NNFLOAT>>()},
                                             gen,
                                             std::vector<std::shared_ptr<BaseLogger<NNFLOAT>>>{
                                                 std::make_shared<CSVLogger<NNFLOAT>>("temp.csv", 999)});

  std::filesystem::create_directory(loc + "/data/nesterov");

  for (size_t i = 0; i < n_networks_train; ++i) {
    auto csv_logger = dynamic_cast<CSVLogger<NNFLOAT> *>(nesterov_training.Loggers()[0].get());
    csv_logger->SetFile(loc + "/data/nesterov/" + std::to_string(i) + ".csv");
    csv_logger->WriteHead();

    nesterov_training.InitializeParameters();
    for (size_t j = 0; j < 1000; ++j)
      nesterov_training.TrainEpoch(inputs, target_outputs, 128, true);
  }

  std::cout << "Nesterov finished" << std::endl;
}

void GenerateData() {
  const std::string loc = PROGRAM_DIR;
  size_t n_networks_train = 100;

  CreateFileSystem(loc);

  auto [inputs, target_outputs] = GenerateSinData(1000);

  std::normal_distribution<NNFLOAT> dist(0, 0.1);
  for (auto &v : target_outputs) {
    v[0] += dist(gen);
  }

  for (auto &v : target_outputs) {
    v[0] = (v[0] + 1) / 2;
  }

  net->Print();

  Adam(inputs, target_outputs, n_networks_train, loc);
  SGD(inputs, target_outputs, n_networks_train, loc);
  RmsProp(inputs, target_outputs, n_networks_train, loc);
  Momentum(inputs, target_outputs, n_networks_train, loc);
  Nesterov(inputs, target_outputs, n_networks_train, loc);
}

int main() {
  GenerateData();

  //test();

  return 0;
}
