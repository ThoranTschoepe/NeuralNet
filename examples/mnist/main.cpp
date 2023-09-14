#include <iostream>

#include <NeuralNet/NeuralNet.h>

#include "mnist/mnist_reader.hpp"

using namespace NeuralNet;
using namespace NeuralNet::Training;

void print_image(const std::vector<uint8_t> &image) {
  for (size_t r = 0; r < 28; ++r) {
    for (size_t c = 0; c < 28; ++c) {
      if (image[r * 28 + c] > 128) {
        std::cout << "#";
      } else if (image[r * 28 + c] > 64) {
        std::cout << "+";
      } else if (image[r * 28 + c] > 32) {
        std::cout << "-";
      } else {
        std::cout << " ";
      }
    }
    std::cout << std::endl;
  }
}

std::shared_ptr<NeuralNetwork<NNFLOAT>> create_network() {
  auto net = std::make_shared<NeuralNetwork<NNFLOAT>>();

  net->AddLayer<FullyConnectedLayer>(784, 256);
  net->AddLayer<SigmoidActivation>(256, 256);
  net->AddLayer<FullyConnectedLayer>(256, 128);
  net->AddLayer<SigmoidActivation>(128, 128);
  net->AddLayer<FullyConnectedLayer>(128, 10);
  net->AddLayer<SoftmaxActivation>(10, 10);

  return net;
}

std::pair<std::vector<std::vector<NNFLOAT>>, std::vector<std::vector<NNFLOAT>>> convert(const std::vector<std::vector<
    uint8_t>> &images, const std::vector<uint8_t> &labels) {
  std::vector<std::vector<NNFLOAT>> converted_images;
  std::vector<std::vector<NNFLOAT>> converted_labels;

  converted_images.reserve(images.size());
  converted_labels.reserve(labels.size());

  for (size_t i = 0; i < images.size(); ++i) {
    converted_images.emplace_back(images[i].begin(), images[i].end());
    converted_labels.emplace_back(10, 0.0f);
    converted_labels[i][labels[i]] = 1.0f;
  }

  return {converted_images, converted_labels};
}

template<std::floating_point dataType>
void test_network(const std::shared_ptr<NeuralNetwork<dataType>> &net,
                  const std::vector<std::vector<dataType>> &images,
                  const std::vector<std::vector<NNFLOAT>> &labels) {
  NetworkInference network_inference(net);
  std::vector<float_t> output(net->OutputSize());

  size_t correct = 0;
  size_t total = images.size();

  for (size_t i = 0; i < total; ++i) {
    output = network_inference(images[i]);

    auto max = std::max_element(output.begin(), output.end());
    auto label = std::distance(output.begin(), max);

    auto max_label = std::max_element(labels[i].begin(), labels[i].end());
    auto correct_label = std::distance(labels[i].begin(), max_label);

    if (label == correct_label) {
      correct++;
    }
  }

  std::cout << "Accuracy: " << correct << "/" << total << " (" << (static_cast<float>(correct) / static_cast<float>(total)) * 100 << "%)"
            << std::endl;
}

void train_network(std::shared_ptr<NeuralNetwork<NNFLOAT>> &net,
                   const std::vector<std::vector<NNFLOAT>> &train_images,
                   const std::vector<std::vector<NNFLOAT>> &train_labels,
                   const std::vector<std::vector<NNFLOAT>> &test_images,
                   const std::vector<std::vector<NNFLOAT>> &test_labels) {
  std::mt19937 gen(std::random_device{}());

  NetworkTraining<NNFLOAT> netTraining(net,
                                       std::make_shared<GradOptimizer<NNFLOAT>>(0.1f),
                                       std::make_shared<CrossEntropyCost<NNFLOAT>>(),
                                       std::vector<std::shared_ptr<BaseInitializer<NNFLOAT>>>{std::make_shared<HeInitializer<NNFLOAT>>()},
                                       gen);

  test_network(net, test_images, test_labels);

  for (size_t i = 0; i < 5; ++i) {
    netTraining.TrainEpoch(train_images, train_labels, 100, true);

    std::cout << "Epoch " << i << std::endl;
    test_network(net, test_images, test_labels);
  }

}

int main() {
  const std::string loc = PROGRAM_DIR;
  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
      mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(loc + "/mnist-master");

  std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
  std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
  std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
  std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

  std::cout << "Sample Image no 35" << std::endl;
  auto i = dataset.training_images[35];

  print_image(i);

  std::cout << "Label: " << dataset.training_labels[35] + 0 << std::endl;

  auto [converted_images, converted_labels] = convert(dataset.training_images, dataset.training_labels);
  auto [converted_test_images, converted_test_labels] = convert(dataset.test_images, dataset.test_labels);

  auto net = create_network();

  train_network(net, converted_images, converted_labels, converted_test_images, converted_test_labels);

  test_network(net, converted_test_images, converted_test_labels);

  return 0;
}
