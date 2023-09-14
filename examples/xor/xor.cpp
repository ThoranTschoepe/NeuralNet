#include <vector>

#include <NeuralNet/NeuralNet.h>

using namespace NeuralNet;
using namespace NeuralNet::Training;

template<std::floating_point dataType>
void TestNetwork(const std::shared_ptr<NeuralNetwork<dataType>> &net, const std::vector<std::vector<float_t>> &inputs) {
  NetworkInference network_inference(net);
  std::vector<dataType> output(net->OutputSize());

  for (const auto &input : inputs) {
    output = network_inference(input);

    for (const auto &i : input) {
      std::cout << i << " ";
    }
    std::cout << "-> ";
    for (const auto &o : output) {
      std::cout << o << " ";
    }
    std::cout << std::endl;
  }
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

std::shared_ptr<NeuralNetwork<NNFLOAT>> CreateSmallNetwork() {
  auto netBuilder = NetworkBuilder<NNFLOAT>(2);

  netBuilder.AddLayer<FullyConnectedLayer>(2, "relu");
  netBuilder.AddLayer<FullyConnectedLayer>(1, "relu");

  auto net = netBuilder.Build();

  return net;
}

std::shared_ptr<NeuralNetwork<NNFLOAT>> CreateMediumNetwork() {
  auto netBuilder = NetworkBuilder<NNFLOAT>(2);

  netBuilder.AddLayer<FullyConnectedLayer>(20, "sigmoid");
  netBuilder.AddLayer<FullyConnectedLayer>(20, "sigmoid");
  netBuilder.AddLayer<FullyConnectedLayer>(1, "sigmoid");

  auto net = netBuilder.Build();

  return net;
}

void SmallNetwork() {
  auto [inputs, targetOutputs] = GenerateXORData();

  auto net = CreateSmallNetwork();

  std::mt19937 gen(std::random_device{}());

  NetworkTraining<NNFLOAT> netTraining(net,
                                       std::make_shared<AdamOptimizer<NNFLOAT>>(0.1f),
                                       std::make_shared<MSECost<NNFLOAT>>(),
                                       std::vector<std::shared_ptr<BaseInitializer<NNFLOAT>>>{
                                           std::make_shared<NormalizedUniformXavierInitializer<NNFLOAT>>()}, //empty arg means all layers
                                       gen);

  NNFLOAT cost = 1;
  size_t iter = 0;
  while (cost > 0.0001f) {
    netTraining.InitializeParameters();
    for (size_t i = 0; i < 1000; i++) {
      netTraining.TrainBatch(inputs, targetOutputs);
    }
    cost = netTraining.CalculateTrainCost();
    ++iter;
  }

  std::cout << "Training finished with cost " << cost << " after " << iter << " iterations" << std::endl;

  std::cout << "Final network:" << std::endl;
  net->Print(std::cout, true);
  std::cout << std::endl;

  std::cout << "Network output after training: " << std::endl;
  TestNetwork(net, inputs);
}

void MediumNetwork() {
  auto [inputs, targetOutputs] = GenerateXORData();

  auto net = CreateMediumNetwork();

  std::mt19937 gen(std::random_device{}());

  //you could just pass a single initializer to the constructor, this is just to show that you can pass multiple
  std::vector<std::shared_ptr<BaseInitializer<NNFLOAT>>> initializers;
  initializers.push_back(make_shared<NormalizedUniformXavierInitializer<NNFLOAT>>(std::vector<size_t>{0}));
  initializers.push_back(make_shared<NormalizedUniformXavierInitializer<NNFLOAT>>(std::vector<size_t>{2}));
  initializers.push_back(make_shared<NormalizedUniformXavierInitializer<NNFLOAT>>(std::vector<size_t>{4}));

  NetworkTraining<NNFLOAT> netTraining(net,
                                       std::make_shared<AdamOptimizer<NNFLOAT>>(0.1f),
                                       std::make_shared<MSECost<NNFLOAT>>(),
                                       initializers,
                                       gen,
                                       std::vector<std::shared_ptr<BaseLogger<NNFLOAT>>>{
                                           std::make_shared<coutLogger<NNFLOAT>>()});

  for (size_t i = 0; i < 1000; i++) {
    netTraining.TrainBatch(inputs, targetOutputs);
  }
  NNFLOAT cost = netTraining.CalculateTrainCost();

  std::cout << "Training finished with cost " << cost << std::endl;

  std::cout << "Network output after training: " << std::endl;
  TestNetwork(net, inputs);
}

int main() {
  SmallNetwork();
  //MediumNetwork();

  return 0;
}
