#pragma once

#include <random>
#include <RTNeural.h>

void randomise_dense(std::unique_ptr<RTNeural::Dense<double>>& dense)
{
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution (-1.0, 1.0);

    // random weights
    std::vector<std::vector<double>> denseWeights (dense->out_size);
    for(auto& w : denseWeights)
        w.resize(dense->in_size, 0.0);

    for(size_t i = 0; i < dense->out_size; ++i)
        for(size_t j = 0; j < dense->in_size; ++j)
            denseWeights[i][j] = distribution(generator);
    
    dense->setWeights(denseWeights);

    // random biases
    std::vector<double> denseBias (dense->out_size);
    for(size_t i = 0; i < dense->out_size; ++i)
        denseBias[i] = distribution(generator);

    dense->setBias(denseBias.data());
}

void randomise_gru(std::unique_ptr<RTNeural::GRULayer<double>>& gru)
{
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0,1.0);

    // kernel weights
    std::vector<std::vector<double>> kernelWeights (gru->in_size);
    for(auto& w : kernelWeights)
        w.resize(3 * gru->out_size, 0.0);

    for(size_t i = 0; i < gru->in_size; ++i)
        for(size_t j = 0; j < 3 * gru->out_size; ++j)
            kernelWeights[i][j] = distribution(generator);
    
    gru->setWVals(kernelWeights);

    // recurrent weights
    std::vector<std::vector<double>> recurrentWeights (gru->out_size);
    for(auto& w : recurrentWeights)
        w.resize(3 * gru->out_size, 0.0);

    for(size_t i = 0; i < gru->out_size; ++i)
        for(size_t j = 0; j < 3 * gru->out_size; ++j)
            recurrentWeights[i][j] = distribution(generator);
    
    gru->setUVals(recurrentWeights);

    // biases
    std::vector<std::vector<double>> gru_bias (2);
    for(auto& w : gru_bias)
        w.resize(3 * gru->out_size, 0.0);

    for(size_t i = 0; i < 2; ++i)
        for(size_t j = 0; j < 3 * gru->out_size; ++j)
            gru_bias[i][j] = distribution(generator);
    
    gru->setBVals(gru_bias);
}

std::unique_ptr<RTNeural::Layer<double>> create_layer(const std::string& layer_type, size_t in_size, size_t out_size)
{
    if(layer_type == "dense")
    {
        auto layer = std::make_unique<RTNeural::Dense<double>>(in_size, out_size);
        randomise_dense(layer);
        return std::move(layer);
    }

    if(layer_type == "gru")
    {
        auto layer = std::make_unique<RTNeural::GRULayer<double>>(in_size, out_size);
        randomise_gru(layer);
        return std::move(layer);
    }

    if(layer_type == "tanh")
    {
        auto layer = std::make_unique<RTNeural::TanhActivation<double>>(in_size);
        return std::move(layer);
    }

    if(layer_type == "relu")
    {
        auto layer = std::make_unique<RTNeural::ReLuActivation<double>>(in_size);
        return std::move(layer);
    }

    std::cout << "Layer type: " << layer_type << " not found!" << std::endl;
    return {};
}
