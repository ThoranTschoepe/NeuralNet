#include <vector>

#include <NeuralNet/NeuralNet.h>

using namespace NeuralNet;
using namespace NeuralNet::Training;

std::pair<std::vector<std::vector<NNFLOAT>>, std::vector<std::vector<NNFLOAT>>> GenerateSinData(size_t count) {
  std::vector<std::vector<NNFLOAT>> inputs;
  std::vector<std::vector<NNFLOAT>> target_outputs;

  inputs.reserve(count);
  target_outputs.reserve(count);

  for (size_t i = 0; i < count; ++i) {
    NNFLOAT x = static_cast<NNFLOAT>(i) * 2 * M_PI / static_cast<NNFLOAT>(count);
    inputs.push_back({x});
    target_outputs.push_back({std::sin(x)});
  }

  return {inputs, target_outputs};
}

std::shared_ptr<NeuralNetwork<NNFLOAT>> CreateNetwork() {
  auto netBuilder = NetworkBuilder<NNFLOAT>(1);

  netBuilder.AddLayer<FullyConnectedLayer>(1, "");
  netBuilder.AddLayer<FullyConnectedLayer>(12, "sigmoid");
  netBuilder.AddLayer<FullyConnectedLayer>(40, "sigmoid");
  netBuilder.AddLayer<FullyConnectedLayer>(12, "sigmoid");
  netBuilder.AddLayer<FullyConnectedLayer>(1, "");

  auto net = netBuilder.Build();

  return net;
}

//save to csv
void toCSV(const std::vector<std::vector<NNFLOAT>> &data, const std::string &filename) {
  std::ofstream file(filename);

  for (const auto &row : data) {
    for (size_t i = 0; i < row.size() - 1; ++i) {
      file << row[i] << ",";
    }
    file << row.back();

    file << "\n";
  }
}

int main() {
  const std::string loc = PROGRAM_DIR;

  auto [inputs, target_outputs] = GenerateSinData(1000);
  auto [test_inputs, test_target_outputs] = GenerateSinData(100);

  std::mt19937 gen(std::random_device{}());
  std::normal_distribution<NNFLOAT> dist(0, 0.1);
  for (auto &v : target_outputs) {
    v[0] += dist(gen);
  }

  toCSV(target_outputs, loc + "/target_outputs.csv");

  for (auto &v : target_outputs) {
    v[0] = (v[0] + 1) / 2;
  }

  for(auto &v : test_target_outputs){
    v[0] = (v[0] + 1) / 2;
  }

  auto net = CreateNetwork();

  NetworkTraining<NNFLOAT> network_training(net,
                                            std::make_shared<AdamOptimizer<NNFLOAT>>(0.01f),
                                            std::make_shared<MSECost<NNFLOAT>>(),
                                            std::vector<std::shared_ptr<BaseInitializer<NNFLOAT>>>{std::make_shared<NormalizedUniformXavierInitializer<NNFLOAT>>()},
                                            gen,
                                            std::vector<std::shared_ptr<BaseLogger<NNFLOAT>>>{std::make_shared<coutLogger<NNFLOAT>>(256*100-1),
                                                                                         std::make_shared<CSVLogger<NNFLOAT>>(loc + "/log.csv", 10)});

  network_training.SetTest(test_inputs, test_target_outputs);

  for (size_t i = 0; i <= 400; i++) {
    network_training.TrainEpoch(inputs, target_outputs, 256, true);
  }

  NetworkInference<NNFLOAT> network_inference(net);

  std::vector<std::vector<NNFLOAT>> outputs;
  outputs.reserve(inputs.size());

  for (const auto &input : inputs) {
    outputs.push_back(network_inference(input));
  }

  for (auto &v : outputs) {
    v[0] = v[0] * 2 - 1;
  }

  toCSV(outputs, loc + "/predicted_outputs.csv");

  return 0;
}
