#pragma once

#include "../modules/json/json.hpp"
#include "Model.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace RTNeural
{
namespace json_parser
{

    /** Creates a dense layer from a json representation of the layer weights */
    template <typename T>
    std::unique_ptr<Dense<T>> createDense(size_t in_size, size_t out_size,
        const nlohmann::json& weights)
    {
        auto dense = std::make_unique<Dense<T>>(in_size, out_size);

        // load weights
        std::vector<std::vector<T>> denseWeights(out_size);
        for(auto& w : denseWeights)
            w.resize(in_size, (T)0);

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

    /** Creates a Conv1D layer from a json representation of the layer weights */
    template <typename T>
    std::unique_ptr<Conv1D<T>> createConv1D(size_t in_size, size_t out_size,
        size_t kernel_size, size_t dilation, const nlohmann::json& weights)
    {
        auto conv = std::make_unique<Conv1D<T>>(in_size, out_size, kernel_size, dilation);

        // load weights
        std::vector<std::vector<std::vector<T>>> convWeights(out_size);
        for(auto& wIn : convWeights)
        {
            wIn.resize(in_size);
            
            for(auto& w : wIn)
                w.resize(kernel_size, (T) 0);
        }

        auto layerWeights = weights[0];
        for(size_t i = 0; i < layerWeights.size(); ++i)
        {
            auto lw = layerWeights[i];
            for(size_t j = 0; j < lw.size(); ++j)
            {
                auto l = lw[j];
                for(size_t k = 0; k < l.size(); ++k)
                    convWeights[k][j][i] = l[k].get<T>();
            }
        }

        conv->setWeights(convWeights);

        // load biases
        std::vector<T> convBias = weights[1].get<std::vector<T>>();
        conv->setBias(convBias);

        return std::move(conv);
    }

    /** Creates a GRU layer from a json representation of the layer weights */
    template <typename T>
    std::unique_ptr<GRULayer<T>> createGRU(size_t in_size, size_t out_size,
        const nlohmann::json& weights)
    {
        auto gru = std::make_unique<GRULayer<T>>(in_size, out_size);

        // load kernel weights
        std::vector<std::vector<T>> kernelWeights(in_size);
        for(auto& w : kernelWeights)
            w.resize(3 * out_size, (T)0);

        auto layerWeights = weights[0];
        for(size_t i = 0; i < layerWeights.size(); ++i)
        {
            auto lw = layerWeights[i];
            for(size_t j = 0; j < lw.size(); ++j)
                kernelWeights[i][j] = lw[j].get<T>();
        }

        gru->setWVals(kernelWeights);

        // load recurrent weights
        std::vector<std::vector<T>> recurrentWeights(out_size);
        for(auto& w : recurrentWeights)
            w.resize(3 * out_size, (T)0);

        auto layerWeights2 = weights[1];
        for(int i = 0; i < layerWeights2.size(); ++i)
        {
            auto lw = layerWeights2[i];
            for(int j = 0; j < lw.size(); ++j)
                recurrentWeights[i][j] = lw[j].get<T>();
        }

        gru->setUVals(recurrentWeights);

        // load biases
        std::vector<std::vector<T>> gruBias(2);
        for(auto& b : gruBias)
            b.resize(3 * out_size, (T)0);

        auto layerBias = weights[2];
        for(int i = 0; i < layerBias.size(); ++i)
        {
            auto lw = layerBias[i];
            for(int j = 0; j < lw.size(); ++j)
                gruBias[i][j] = lw[j].get<T>();
        }

        gru->setBVals(gruBias);

        return std::move(gru);
    }

    /** Creates a LSTM layer from a json representation of the layer weights */
    template <typename T>
    std::unique_ptr<LSTMLayer<T>> createLSTM(size_t in_size, size_t out_size,
        const nlohmann::json& weights)
    {
        auto lstm = std::make_unique<LSTMLayer<T>>(in_size, out_size);

        // load kernel weights
        std::vector<std::vector<T>> kernelWeights(in_size);
        for(auto& w : kernelWeights)
            w.resize(4 * out_size, (T)0);

        auto layerWeights = weights[0];
        for(size_t i = 0; i < layerWeights.size(); ++i)
        {
            auto lw = layerWeights[i];
            for(size_t j = 0; j < lw.size(); ++j)
                kernelWeights[i][j] = lw[j].get<T>();
        }

        lstm->setWVals(kernelWeights);

        // load recurrent weights
        std::vector<std::vector<T>> recurrentWeights(out_size);
        for(auto& w : recurrentWeights)
            w.resize(4 * out_size, (T)0);

        auto layerWeights2 = weights[1];
        for(int i = 0; i < layerWeights2.size(); ++i)
        {
            auto lw = layerWeights2[i];
            for(int j = 0; j < lw.size(); ++j)
                recurrentWeights[i][j] = lw[j].get<T>();
        }

        lstm->setUVals(recurrentWeights);

        // load biases
        std::vector<T> lstmBias = weights[2].get<std::vector<T>>();
        lstm->setBVals(lstmBias);

        return std::move(lstm);
    }

    /** Creates an activation layer of a given type */
    template <typename T>
    std::unique_ptr<Activation<T>>
    createActivation(const std::string& activationType, size_t dims)
    {
        if(activationType == "tanh")
            return std::make_unique<TanhActivation<T>>(dims);

        if(activationType == "relu")
            return std::make_unique<ReLuActivation<T>>(dims);

        if(activationType == "sigmoid")
            return std::make_unique<SigmoidActivation<T>>(dims);

        return {};
    }

    static void debug_print(std::string str, bool debug)
    {
        if(debug)
            std::cout << str << std::endl;
    }

    /** Creates a neural network model from a json stream */
    template <typename T>
    std::unique_ptr<Model<T>> parseJson(const nlohmann::json& parent, bool debug = false)
    {
        auto shape = parent["in_shape"];
        auto layers = parent["layers"];

        if(!shape.is_array() || !layers.is_array())
            return {};

        const auto nDims = shape.back().get<int>();
        debug_print("# dimensions: " + std::to_string(nDims), debug);

        auto model = std::make_unique<Model<T>>(nDims);

        for(const auto& l : layers)
        {
            const auto type = l["type"].get<std::string>();
            debug_print("Layer: " + type, debug);

            const auto layerShape = l["shape"];
            const auto layerDims = layerShape.back().get<int>();
            debug_print("  Dims: " + std::to_string(layerDims), debug);

            const auto weights = l["weights"];

            auto add_activation = [=](std::unique_ptr<Model<T>>& model, const nlohmann::json& l)
            {
                if(l.contains("activation"))
                {
                    const auto activationType = l["activation"].get<std::string>();
                    if(!activationType.empty())
                    {
                        debug_print("  activation: " + activationType, debug);
                        auto activation = createActivation<T>(activationType, layerDims);
                        model->addLayer(activation.release());
                    }
                }
            };

            if(type == "dense" || type == "time-distributed-dense")
            {
                auto dense = createDense<T>(model->getNextInSize(), layerDims, weights);
                model->addLayer(dense.release());
                add_activation(model, l);
            }
            else if(type == "conv1d")
            {
                const auto kernel_size = l["kernel_size"].back().get<int>();
                const auto dilation = l["dilation"].back().get<int>();

                auto conv = createConv1D<T>(model->getNextInSize(), layerDims, kernel_size, dilation, weights);
                model->addLayer(conv.release());
                add_activation(model, l);
            }
            else if(type == "gru")
            {
                auto gru = createGRU<T>(model->getNextInSize(), layerDims, weights);
                model->addLayer(gru.release());
            }
            else if(type == "lstm")
            {
                auto lstm = createLSTM<T>(model->getNextInSize(), layerDims, weights);
                model->addLayer(lstm.release());
            }
        }

        return std::move(model);
    }

    /** Creates a neural network model from a json stream */
    template <typename T>
    std::unique_ptr<Model<T>> parseJson(std::ifstream& jsonStream, bool debug = false)
    {
        nlohmann::json parent;
        jsonStream >> parent;
        return parseJson<T>(parent, debug);
    }

} // namespace json_parser
} // namespace RTNeural
