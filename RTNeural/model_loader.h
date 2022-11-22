#pragma once

#include "../modules/json/json.hpp"
#include "Model.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace RTNeural
{
/** Utility functions for loading model weights from their json representation. */
namespace json_parser
{
    [[maybe_unused]] static void debug_print(const std::string& str, bool debug)
    {
        if(debug)
            std::cout << str << std::endl;
    }

    /** Loads weights for a Dense (or DenseT) layer from a json representation of the layer weights. */
    template <typename T, typename DenseType>
    void loadDense(DenseType& dense, const nlohmann::json& weights)
    {
        // load weights
        std::vector<std::vector<T>> denseWeights(dense.out_size);
        for(auto& w : denseWeights)
            w.resize(dense.in_size, (T)0);

        auto layerWeights = weights.at(0);
        for(size_t i = 0; i < layerWeights.size(); ++i)
        {
            auto lw = layerWeights.at(i);
            for(size_t j = 0; j < lw.size(); ++j)
                denseWeights.at(j).at(i) = lw.at(j).get<T>();
        }

        dense.setWeights(denseWeights);

        // load biases
        std::vector<T> denseBias = weights.at(1).get<std::vector<T>>();
        dense.setBias(denseBias.data());
    }

    /** Creates a Dense layer from a json representation of the layer weights. */
    template <typename T>
    std::unique_ptr<Dense<T>> createDense(int in_size, int out_size, const nlohmann::json& weights)
    {
        auto dense = std::make_unique<Dense<T>>(in_size, out_size);
        loadDense<T>(*dense.get(), weights);
        return std::move(dense);
    }

    /** Checks that a Dense (or DenseT) layer has the given dimensions. */
    template <typename T, typename DenseType>
    bool checkDense(const DenseType& dense, const std::string& type, int layerDims, const bool debug)
    {
        if(type != "dense" && type != "time-distributed-dense")
        {
            debug_print("Wrong layer type! Expected: Dense", debug);
            return false;
        }

        if(layerDims != dense.out_size)
        {
            debug_print("Wrong layer size! Expected: " + std::to_string(dense.out_size), debug);
            return false;
        }

        return true;
    }

    /** Loads weights for a Conv1D (or Conv1DT) layer from a json representation of the layer weights. */
    template <typename T, typename Conv1DType>
    void loadConv1D(Conv1DType& conv, int kernel_size, int /*dilation*/, const nlohmann::json& weights)
    {
        // load weights
        std::vector<std::vector<std::vector<T>>> convWeights(conv.out_size);
        for(auto& wIn : convWeights)
        {
            wIn.resize(conv.in_size);

            for(auto& w : wIn)
                w.resize(kernel_size, (T)0);
        }

        auto layerWeights = weights.at(0);
        for(size_t i = 0; i < layerWeights.size(); ++i)
        {
            auto lw = layerWeights.at(i);
            for(size_t j = 0; j < lw.size(); ++j)
            {
                auto l = lw.at(j);
                for(size_t k = 0; k < l.size(); ++k)
                    convWeights.at(k).at(j).at(kernel_size - 1 - i) = l.at(k).get<T>();
            }
        }

        conv.setWeights(convWeights);

        // load biases
        std::vector<T> convBias = weights.at(1).get<std::vector<T>>();
        conv.setBias(convBias);
    }

    /** Creates a Conv1D layer from a json representation of the layer weights. */
    template <typename T>
    std::unique_ptr<Conv1D<T>> createConv1D(int in_size, int out_size,
        int kernel_size, int dilation, const nlohmann::json& weights)
    {
        auto conv = std::make_unique<Conv1D<T>>(in_size, out_size, kernel_size, dilation);
        loadConv1D<T>(*conv.get(), kernel_size, dilation, weights);
        return std::move(conv);
    }

    /** Checks that a Conv1D (or Conv1DT) layer has the given dimensions. */
    template <typename T, typename Conv1DType>
    bool checkConv1D(const Conv1DType& conv, const std::string& type, int layerDims,
        int kernel_size, int dilation_rate, const bool debug)
    {
        if(type != "conv1d")
        {
            debug_print("Wrong layer type! Expected: Conv1D", debug);
            return false;
        }

        if(layerDims != conv.out_size)
        {
            debug_print("Wrong layer size! Expected: " + std::to_string(conv.out_size), debug);
            return false;
        }

        if(kernel_size != conv.getKernelSize())
        {
            debug_print("Wrong kernel size! Expected: " + std::to_string(conv.getKernelSize()), debug);
            return false;
        }

        if(dilation_rate != conv.getDilationRate())
        {
            debug_print("Wrong dilation_rate! Expected: " + std::to_string(conv.getDilationRate()), debug);
            return false;
        }

        return true;
    }

    /** Loads weights for a GRULayer (or GRULayerT) from a json representation of the layer weights. */
    template <typename T, typename GRUType>
    void loadGRU(GRUType& gru, const nlohmann::json& weights)
    {
        // load kernel weights
        std::vector<std::vector<T>> kernelWeights(gru.in_size);
        for(auto& w : kernelWeights)
            w.resize(3 * gru.out_size, (T)0);

        auto layerWeights = weights.at(0);
        for(size_t i = 0; i < layerWeights.size(); ++i)
        {
            auto lw = layerWeights.at(i);
            for(size_t j = 0; j < lw.size(); ++j)
                kernelWeights.at(i).at(j) = lw.at(j).get<T>();
        }

        gru.setWVals(kernelWeights);

        // load recurrent weights
        std::vector<std::vector<T>> recurrentWeights(gru.out_size);
        for(auto& w : recurrentWeights)
            w.resize(3 * gru.out_size, (T)0);

        auto layerWeights2 = weights.at(1);
        for(size_t i = 0; i < layerWeights2.size(); ++i)
        {
            auto lw = layerWeights2.at(i);
            for(size_t j = 0; j < lw.size(); ++j)
                recurrentWeights.at(i).at(j) = lw.at(j).get<T>();
        }

        gru.setUVals(recurrentWeights);

        // load biases
        std::vector<std::vector<T>> gruBias(2);
        for(auto& b : gruBias)
            b.resize(3 * gru.out_size, (T)0);

        auto layerBias = weights.at(2);
        for(size_t i = 0; i < layerBias.size(); ++i)
        {
            auto lw = layerBias.at(i);
            for(size_t j = 0; j < lw.size(); ++j)
                gruBias.at(i).at(j) = lw.at(j).get<T>();
        }

        gru.setBVals(gruBias);
    }

    /** Creates a GRULayer from a json representation of the layer weights. */
    template <typename T>
    std::unique_ptr<GRULayer<T>> createGRU(int in_size, int out_size, const nlohmann::json& weights)
    {
        auto gru = std::make_unique<GRULayer<T>>(in_size, out_size);
        loadGRU<T>(*gru.get(), weights);
        return std::move(gru);
    }

    /** Checks that a GRULayer (or GRULayerT) has the given dimensions. */
    template <typename T, typename GRUType>
    bool checkGRU(const GRUType& gru, const std::string& type, int layerDims, const bool debug)
    {
        if(type != "gru")
        {
            debug_print("Wrong layer type! Expected: GRU", debug);
            return false;
        }

        if(layerDims != gru.out_size)
        {
            debug_print("Wrong layer size! Expected: " + std::to_string(gru.out_size), debug);
            return false;
        }

        return true;
    }

    /** Loads weights for a LSTMLayer (or LSTMLayerT) from a json representation of the layer weights. */
    template <typename T, typename LSTMType>
    void loadLSTM(LSTMType& lstm, const nlohmann::json& weights)
    {
        // load kernel weights
        std::vector<std::vector<T>> kernelWeights(lstm.in_size);
        for(auto& w : kernelWeights)
            w.resize(4 * lstm.out_size, (T)0);

        auto layerWeights = weights.at(0);
        for(size_t i = 0; i < layerWeights.size(); ++i)
        {
            auto lw = layerWeights.at(i);
            for(size_t j = 0; j < lw.size(); ++j)
                kernelWeights.at(i).at(j) = lw.at(j).get<T>();
        }

        lstm.setWVals(kernelWeights);

        // load recurrent weights
        std::vector<std::vector<T>> recurrentWeights(lstm.out_size);
        for(auto& w : recurrentWeights)
            w.resize(4 * lstm.out_size, (T)0);

        auto layerWeights2 = weights.at(1);
        for(size_t i = 0; i < layerWeights2.size(); ++i)
        {
            auto lw = layerWeights2.at(i);
            for(size_t j = 0; j < lw.size(); ++j)
                recurrentWeights.at(i).at(j) = lw.at(j).get<T>();
        }

        lstm.setUVals(recurrentWeights);

        // load biases
        std::vector<T> lstmBias = weights.at(2).get<std::vector<T>>();
        lstm.setBVals(lstmBias);
    }

    /** Creates a LSTMLayer from a json representation of the layer weights. */
    template <typename T>
    std::unique_ptr<LSTMLayer<T>> createLSTM(int in_size, int out_size, const nlohmann::json& weights)
    {
        auto lstm = std::make_unique<LSTMLayer<T>>(in_size, out_size);
        loadLSTM<T>(*lstm.get(), weights);
        return std::move(lstm);
    }

    /** Checks that a LSTMLayer (or LSTMLayerT) has the given dimensions. */
    template <typename T, typename LSTMType>
    bool checkLSTM(const LSTMType& lstm, const std::string& type, int layerDims, const bool debug)
    {
        if(type != "lstm")
        {
            debug_print("Wrong layer type! Expected: LSTM", debug);
            return false;
        }

        if(layerDims != lstm.out_size)
        {
            debug_print("Wrong layer size! Expected: " + std::to_string(lstm.out_size), debug);
            return false;
        }

        return true;
    }

    /** Creates an activation layer of a given type. */
    template <typename T>
    std::unique_ptr<Activation<T>>
    createActivation(const std::string& activationType, int dims)
    {
        if(activationType == "tanh")
            return std::make_unique<TanhActivation<T>>(dims);

        if(activationType == "relu")
            return std::make_unique<ReLuActivation<T>>(dims);

        if(activationType == "sigmoid")
            return std::make_unique<SigmoidActivation<T>>(dims);

        if(activationType == "softmax")
            return std::make_unique<SoftmaxActivation<T>>(dims);

        if(activationType == "elu")
            return std::make_unique<ELuActivation<T>>(dims);

        return {};
    }

    /** Checks that an Activation layer has the given dimensions */
    template <typename LayerType>
    bool checkActivation(const LayerType& actLayer, const std::string& activationType, int dims, const bool debug)
    {
        if(dims != actLayer.out_size)
        {
            debug_print("Wrong layer size! Expected: " + std::to_string(actLayer.out_size), debug);
            return false;
        }

        if(activationType != actLayer.getName())
        {
            debug_print("Wrong layer type! Expected: " + actLayer.getName(), debug);
            return false;
        }

        return true;
    }

    /** Creates a neural network model from a json stream. */
    template <typename T>
    std::unique_ptr<Model<T>> parseJson(const nlohmann::json& parent, const bool debug = false)
    {
        auto shape = parent.at("in_shape");
        auto layers = parent.at("layers");

        if(!shape.is_array() || !layers.is_array())
            return {};

        const auto nDims = shape.back().get<int>();
        debug_print("# dimensions: " + std::to_string(nDims), debug);

        auto model = std::make_unique<Model<T>>(nDims);

        for(const auto& l : layers)
        {
            const auto type = l.at("type").get<std::string>();
            debug_print("Layer: " + type, debug);

            const auto layerShape = l.at("shape");
            const auto layerDims = layerShape.back().get<int>();
            debug_print("  Dims: " + std::to_string(layerDims), debug);

            const auto weights = l.at("weights");

            auto add_activation = [=](std::unique_ptr<Model<T>>& _model, const nlohmann::json& _l) {
                if(_l.contains("activation"))
                {
                    const auto activationType = _l["activation"].get<std::string>();
                    if(!activationType.empty())
                    {
                        debug_print("  activation: " + activationType, debug);
                        auto activation = createActivation<T>(activationType, layerDims);
                        _model->addLayer(activation.release());
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
                const auto kernel_size = l.at("kernel_size").back().get<int>();
                const auto dilation = l.at("dilation").back().get<int>();

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

    /** Creates a neural network model from a json stream. */
    template <typename T>
    std::unique_ptr<Model<T>> parseJson(std::ifstream& jsonStream, const bool debug = false)
    {
        nlohmann::json parent;
        jsonStream >> parent;
        return parseJson<T>(parent, debug);
    }

} // namespace json_parser
} // namespace RTNeural
