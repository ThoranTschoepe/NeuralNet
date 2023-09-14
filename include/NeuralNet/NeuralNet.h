#pragma once

#include <NeuralNet/misc/types.h>
#include <NeuralNet/misc/layer_type.h>
#include <NeuralNet/misc/network_builder.h>


/* Classes used for artificial neural networks inference */

#include <NeuralNet/Layers/fully_connected_layer.h>

#include <NeuralNet/Layers/Activations/tanh_activation.h>
#include <NeuralNet/Layers/Activations/relu_activation.h>
#include <NeuralNet/Layers/Activations/leaky_relu_activation.h>
#include <NeuralNet/Layers/Activations/softmax_activation.h>
#include <NeuralNet/Layers/Activations/sigmoid_activation.h>

#include <NeuralNet/Model/base_network.h>
#include <NeuralNet/Model/inference_network.h>

/* Classes used for artificial neural networks training */

#include <NeuralNet/CostFunctions/MSE_cost.h>
#include <NeuralNet/CostFunctions/absolute_cost.h>
#include <NeuralNet/CostFunctions/cross_entropy_cost.h>

#include <NeuralNet/Loggers/cout_logger.h>
#include <NeuralNet/Loggers/csv_logger.h>

#include <NeuralNet/Optimizers/grad_optimizer.h>
#include <NeuralNet/Optimizers/adam_optimizer.h>
#include <NeuralNet/Optimizers/momentum_optimizer.h>
#include <NeuralNet/Optimizers/rmsprop_optimizer.h>
#include <NeuralNet/Optimizers/nesterov_optimizer.h>

#include <NeuralNet/Model/training_network.h>

#include <NeuralNet/Initializers/xavier_initializer.h>
#include <NeuralNet/Initializers/he_initializer.h>