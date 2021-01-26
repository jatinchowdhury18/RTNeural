#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include "Model.h"
#include "../modules/json/json.hpp"

namespace RTNeural {
namespace json_parser {

/** Creates a dense layer from a json representation of the layer weights */
template<typename T>
std::unique_ptr<Dense<T>> createDense(size_t in_size, size_t out_size, const nlohmann::json& weights)
{
    auto dense = std::make_unique<Dense<T>>(in_size, out_size);

    // load weights
    std::vector<std::vector<T>> denseWeights (out_size);
    for(auto& w : denseWeights)
        w.resize(in_size, (T) 0);

    auto layerWeights = weights[0];
    for(size_t i = 0; i < layerWeights.size(); ++i)
    {
        auto lw = layerWeights[i];
        for(size_t j = 0; j < lw.size(); ++j)
            denseWeights[j][i] = lw[j].get<T>();
    }
    
    dense->setWeights(denseWeights);

    // load biases
    std::vector<T> denseBias = weights[1].get<std::vector<T>>();
    dense->setBias(denseBias.data());

    return std::move(dense);
}

/** Creates a GRU layer from a json representation of the layer weights */
template<typename T>
std::unique_ptr<GRULayer<T>> createGRU (size_t in_size, size_t out_size, const nlohmann::json& weights)
{
    auto gru = std::make_unique<GRULayer<T>> (in_size, out_size);

    // load kernel weights
    std::vector<std::vector<T>> kernelWeights (in_size);
    for(auto& w : kernelWeights)
        w.resize(3 * out_size, (T) 0);

    auto layerWeights = weights[0];
    for(size_t i = 0; i < layerWeights.size(); ++i)
    {
        auto lw = layerWeights[i];
        for(size_t j = 0; j < lw.size(); ++j)
            kernelWeights[i][j] = lw[j].get<T>();
    }

    gru->setWVals (kernelWeights);

    // load recurrent weights
    std::vector<std::vector<T>> recurrentWeights (out_size);
    for(auto& w : recurrentWeights)
        w.resize(3 * out_size, (T) 0);
        
    auto layerWeights2 = weights[1];
    for (int i = 0; i < layerWeights2.size(); ++i)
    {
        auto lw = layerWeights2[i];
        for (int j = 0; j < lw.size(); ++j)
            recurrentWeights[i][j] = lw[j].get<T>();
    }

    gru->setUVals (recurrentWeights);

    // load biases
    std::vector<std::vector<T>> gruBias (2);
    for(auto& b : gruBias)
        b.resize(3 * out_size, (T) 0);
        
    auto layerBias = weights[2];
    for (int i = 0; i < layerBias.size(); ++i)
    {
        auto lw = layerBias[i];
        for (int j = 0; j < lw.size(); ++j)
            gruBias[i][j] = lw[j].get<T>();
    }

    gru->setBVals (gruBias);

    return std::move (gru);
}

/** Creates an activation layer of a given type */
template<typename T>
std::unique_ptr<Activation<T>> createActivation (const std::string& activationType, size_t dims)
{
    if (activationType == "tanh")
        return std::make_unique<TanhActivation<T>>(dims);

    if (activationType == "relu")
        return std::make_unique<ReLuActivation<T>>(dims);

    if (activationType == "sigmoid")
        return std::make_unique<SigmoidActivation<T>>(dims);

    return {};
}

/** Creates a neural network model from a json stream */
template<typename T>
std::unique_ptr<Model<T>> parseJson (std::ifstream& jsonStream)
{
    nlohmann::json parent;
    jsonStream >> parent;
    auto shape = parent["in_shape"];
    auto layers = parent["layers"];

    if(! shape.is_array() || ! layers.is_array())
        return {};

    const auto nDims = shape.back().get<int>();
    std::cout << "# dimensions: " << nDims << std::endl;

    auto model = std::make_unique<Model<T>>(nDims);

    for(const auto& l : layers)
    {
        const auto type = l["type"].get<std::string>();
        std::cout << "Layer: " << type << std::endl;

        const auto layerShape = l["shape"];
        const auto layerDims = layerShape.back().get<int>();
        std::cout << "  Dims: " << layerDims << std::endl;

        const auto weights = l["weights"];

        if(type == "dense" || type == "time-distributed-dense")
        {
            auto dense = createDense<T>(model->getNextInSize(), layerDims, weights);
            model->addLayer(dense.release());

            const auto activationType = l["activation"].get<std::string>();
            if (! activationType.empty())
            {
                std::cout << "  activation: " << activationType << std::endl;
                auto activation = createActivation<T>(activationType, layerDims);
                model->addLayer(activation.release());
            }
        }
        else if(type == "gru")
        {
            auto gru = createGRU<T>(model->getNextInSize(), layerDims, weights);
            model->addLayer(gru.release());
        }
    }

    return std::move(model);
}

} // namespace json_parser
} // namespace RTNeural
