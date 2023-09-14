#include <NeuralNet/NeuralNet.h>

#include <filesystem>

using namespace NeuralNet;
using namespace NeuralNet::Training;

std::shared_ptr<NeuralNetwork<NNFLOAT>> CreateNetwork() {
  auto netBuilder = NetworkBuilder<NNFLOAT>(2);

  netBuilder.AddLayer<FullyConnectedLayer>(20, "sigmoid");
  netBuilder.AddLayer<FullyConnectedLayer>(20, "sigmoid");
  netBuilder.AddLayer<FullyConnectedLayer>(1, "sigmoid");

  auto net = netBuilder.Build();

  return net;
}

template<typename T>
bool compareFloats(T a, T b, T epsilon = 0.0001) {
  return (a - b) < epsilon && (b - a) < epsilon;
}

bool compareWeights(const std::shared_ptr<NeuralNetwork<NNFLOAT>> &net1,
                    const std::shared_ptr<NeuralNetwork<NNFLOAT>> &net2) {
  for (int i = 0; i < net1->LayersSize(); ++i) {
    auto layer1 = net1->LayerAt(i);
    auto layer2 = net2->LayerAt(i);

    if (auto lc1 = dynamic_pointer_cast<FullyConnectedLayer<NNFLOAT>>(layer1)) {
      auto lc2 = dynamic_pointer_cast<FullyConnectedLayer<NNFLOAT>>(layer2);
      for (int j = 0; j < lc1->ParametersSize(); ++j) {
        if (!compareFloats(lc1->Parameters()[j], lc2->Parameters()[j])) {
          return false;
        }
      }
    }
  }

  return true;
}

std::pair<std::vector<std::vector<NNFLOAT>>, std::vector<std::vector<NNFLOAT>>> GenerateXORData() {
  std::vector<std::vector<NNFLOAT>> inputs;
  std::vector<std::vector<NNFLOAT>> target_outputs;

  inputs.push_back({0, 0});
  inputs.push_back({1, 0});
  inputs.push_back({0, 1});
  inputs.push_back({1, 1});

  target_outputs.push_back({0});
  target_outputs.push_back({1});
  target_outputs.push_back({1});
  target_outputs.push_back({0});

  return {inputs, target_outputs};
}

void train(NetworkTraining<NNFLOAT> &network_training) {
  auto [inputs, targetOutputs] = GenerateXORData();

  for (size_t i = 0; i < 500; i++) {
    network_training.TrainBatch(inputs, targetOutputs);
  }
}

int main() {
  auto original_net = CreateNetwork();

  std::mt19937 gen(std::random_device{}());

  NetworkTraining<NNFLOAT> original_net_training(original_net,
                                                 std::make_shared<AdamOptimizer<NNFLOAT>>(0.1f),
                                                 std::make_shared<MSECost<NNFLOAT>>(),
                                                 std::vector<std::shared_ptr<BaseInitializer<NNFLOAT>>>{
                                                     std::make_shared<NormalizedUniformXavierInitializer<NNFLOAT>>()}, //empty arg means all layers
                                                 gen,
                                                 std::vector<std::shared_ptr<BaseLogger<NNFLOAT>>>{
                                                     std::make_shared<coutLogger<NNFLOAT>>()},
                                                 true);

  const std::string loc = PROGRAM_DIR;

  original_net->Save(loc + "/test.nn");

  auto net2 = Load(loc + "/test.nn");

  NetworkTraining<NNFLOAT> net2_training(net2,
                                         std::make_shared<AdamOptimizer<NNFLOAT>>(0.1f),
                                         std::make_shared<MSECost<NNFLOAT>>(),
                                         std::vector<std::shared_ptr<BaseInitializer<NNFLOAT>>>{
                                             std::make_shared<NormalizedUniformXavierInitializer<NNFLOAT>>()}, //empty arg means all layers
                                         gen,
                                         std::vector<std::shared_ptr<BaseLogger<NNFLOAT>>>{
                                             std::make_shared<coutLogger<NNFLOAT>>(),
                                             std::make_shared<CSVLogger<NNFLOAT>>(loc + "/test.csv")},
                                         false);

  std::filesystem::remove(loc + "/test.nn");

  std::cout << compareWeights(original_net, net2) << std::endl;

  train(original_net_training);
  std::cout << std::endl;
  train(net2_training);

  std::cout << compareWeights(original_net, net2) << std::endl;

  return 0;
}
