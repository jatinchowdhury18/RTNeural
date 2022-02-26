#pragma once

#include <RTNeural.h>
#include <random>

template <typename DenseType>
void randomise_dense(DenseType &dense) {
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  // random weights
  std::vector<std::vector<double>> denseWeights(dense.out_size);
  for (auto &w : denseWeights)
    w.resize(dense.in_size, 0.0);

  for (size_t i = 0; i < dense.out_size; ++i)
    for (size_t j = 0; j < dense.in_size; ++j)
      denseWeights[i][j] = distribution(generator);

  dense.setWeights(denseWeights);

  // random biases
  std::vector<double> denseBias(dense.out_size);
  for (size_t i = 0; i < dense.out_size; ++i)
    denseBias[i] = distribution(generator);

  dense.setBias(denseBias.data());
}

template <typename ConvType>
void randomise_conv1d(ConvType &conv, size_t kernel_size) {
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  // random weights
  std::vector<std::vector<std::vector<double>>> convWeights(conv.out_size);
  for(auto& wIn : convWeights)
  {
      wIn.resize(conv.in_size);
      for(auto& w : wIn)
          w.resize(kernel_size, 0.0);
  }

  for (size_t i = 0; i < conv.out_size; ++i)
    for (size_t j = 0; j < conv.in_size; ++j)
      for (size_t k = 0; k < kernel_size; ++k)
        convWeights[i][j][k] = distribution(generator);

  conv.setWeights(convWeights);

  // random biases
  std::vector<double> convBias(conv.out_size);
  for (size_t i = 0; i < conv.out_size; ++i)
    convBias[i] = distribution(generator);

  conv.setBias(convBias);
}

template <typename GruType>
void randomise_gru(GruType &gru) {
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  // kernel weights
  std::vector<std::vector<double>> kernelWeights(gru.in_size);
  for (auto &w : kernelWeights)
    w.resize(3 * gru.out_size, 0.0);

  for (size_t i = 0; i < gru.in_size; ++i)
    for (size_t j = 0; j < 3 * gru.out_size; ++j)
      kernelWeights[i][j] = distribution(generator);

  gru.setWVals(kernelWeights);

  // recurrent weights
  std::vector<std::vector<double>> recurrentWeights(gru.out_size);
  for (auto &w : recurrentWeights)
    w.resize(3 * gru.out_size, 0.0);

  for (size_t i = 0; i < gru.out_size; ++i)
    for (size_t j = 0; j < 3 * gru.out_size; ++j)
      recurrentWeights[i][j] = distribution(generator);

  gru.setUVals(recurrentWeights);

  // biases
  std::vector<std::vector<double>> gru_bias(2);
  for (auto &w : gru_bias)
    w.resize(3 * gru.out_size, 0.0);

  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 3 * gru.out_size; ++j)
      gru_bias[i][j] = distribution(generator);

  gru.setBVals(gru_bias);
}

template <typename LstmType>
void randomise_lstm(LstmType &lstm) {
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  // kernel weights
  std::vector<std::vector<double>> kernelWeights(lstm.in_size);
  for (auto &w : kernelWeights)
    w.resize(4 * lstm.out_size, 0.0);

  for (size_t i = 0; i < lstm.in_size; ++i)
    for (size_t j = 0; j < 4 * lstm.out_size; ++j)
      kernelWeights[i][j] = distribution(generator);

  lstm.setWVals(kernelWeights);

  // recurrent weights
  std::vector<std::vector<double>> recurrentWeights(lstm.out_size);
  for (auto &w : recurrentWeights)
    w.resize(4 * lstm.out_size, 0.0);

  for (size_t i = 0; i < lstm.out_size; ++i)
    for (size_t j = 0; j < 4 * lstm.out_size; ++j)
      recurrentWeights[i][j] = distribution(generator);

  lstm.setUVals(recurrentWeights);

  // biases
  std::vector<double> lstm_bias(4 * lstm.out_size);
  for (size_t i = 0; i < 4 * lstm.out_size; ++i)
    lstm_bias[i] = distribution(generator);

  lstm.setBVals(lstm_bias);
}

std::unique_ptr<RTNeural::Layer<double>>
create_layer(const std::string &layer_type, size_t in_size, size_t out_size) {
  if (layer_type == "dense") {
    auto layer = std::make_unique<RTNeural::Dense<double>>(in_size, out_size);
    randomise_dense(*layer);
    return std::move(layer);
  }

  if (layer_type == "conv1d") {
    const auto kernel_size = in_size - 1;
    auto layer = std::make_unique<RTNeural::Conv1D<double>>(in_size, out_size, kernel_size, 1);
    randomise_conv1d(*layer, kernel_size);
    return std::move(layer);
  }

  if (layer_type == "gru") {
    auto layer =
        std::make_unique<RTNeural::GRULayer<double>>(in_size, out_size);
    randomise_gru(*layer);
    return std::move(layer);
  }

  if (layer_type == "lstm") {
    auto layer =
        std::make_unique<RTNeural::LSTMLayer<double>>(in_size, out_size);
    randomise_lstm(*layer);
    return std::move(layer);
  }

  if (layer_type == "tanh") {
    auto layer = std::make_unique<RTNeural::TanhActivation<double>>(in_size);
    return std::move(layer);
  }

  if (layer_type == "fast_tanh") {
    auto layer = std::make_unique<RTNeural::FastTanh<double>>(in_size);
    return std::move(layer);
  }

  if (layer_type == "relu") {
    auto layer = std::make_unique<RTNeural::ReLuActivation<double>>(in_size);
    return std::move(layer);
  }

  if (layer_type == "sigmoid") {
    auto layer = std::make_unique<RTNeural::SigmoidActivation<double>>(in_size);
    return std::move(layer);
  }

  if (layer_type == "softmax") {
    auto layer = std::make_unique<RTNeural::SoftmaxActivation<double>>(in_size);
    return std::move(layer);
  }

  std::cout << "Layer type: " << layer_type << " not found!" << std::endl;
  return {};
}
