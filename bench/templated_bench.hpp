#pragma once

#include "bench_utils.hpp"
#include "layer_creator.hpp"
#include <chrono>
#include <RTNeural.h>

#if MODELT_AVAILABLE

double runTemplatedBench(const std::vector<vec_type>& signal, const size_t n_samples,
    const std::string& layer_type, size_t in_size, size_t out_size)
{
    using namespace RTNeural;
    using clock_t = std::chrono::high_resolution_clock;
    using second_t = std::chrono::duration<double>;

    auto run_layer = [=, &signal] (auto& layer) -> double
    {
        auto start = clock_t::now();
        for(size_t i = 0; i < n_samples; ++i)
            layer.forward(signal[i].data());
        return std::chrono::duration_cast<second_t>(clock_t::now() - start).count();
    };

    double duration = 10000.0;
    if(layer_type == "dense")
    {
        if(in_size == 4 && out_size == 4)
        {
            ModelT<double, 4, 4, DenseT<double, 4, 4>> model;
            randomise_dense (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 8 && out_size == 8)
        {
            ModelT<double, 8, 8, DenseT<double, 8, 8>> model;
            randomise_dense (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 16 && out_size == 16)
        {
            ModelT<double, 16, 16, DenseT<double, 16, 16>> model;
            randomise_dense (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 24 && out_size == 1)
        {
            ModelT<double, 24, 1, DenseT<double, 24, 1>> model;
            randomise_dense (model.get<0>());
            duration = run_layer(model);
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }
    else if(layer_type == "conv1d")
    {
        if(in_size == 4 && out_size == 4)
        {
            constexpr size_t kernel_size = 3; // in_size - 1
            ModelT<double, 4, 4, Conv1DT<double, 4, 4, kernel_size, 1>> model;
            randomise_conv1d (model.get<0>(), kernel_size);
            duration = run_layer(model);
        }
        else if(in_size == 8 && out_size == 8)
        {
            constexpr size_t kernel_size = 7; // in_size - 1
            constexpr size_t dilation_rate = 2; // in_size / 4
            ModelT<double, 8, 8, Conv1DT<double, 8, 8, kernel_size, dilation_rate>> model;
            randomise_conv1d (model.get<0>(), kernel_size);
            duration = run_layer(model);
        }
        else if(in_size == 16 && out_size == 16)
        {
            constexpr size_t kernel_size = 15; // in_size - 1
            constexpr size_t dilation_rate = 4; // in_size / 4
            ModelT<double, 16, 16, Conv1DT<double, 16, 16, kernel_size, dilation_rate>> model;
            randomise_conv1d (model.get<0>(), kernel_size);
            duration = run_layer(model);
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }
    else if(layer_type == "gru")
    {
        if(in_size == 4 && out_size == 4)
        {
            ModelT<double, 4, 4, GRULayerT<double, 4, 4>> model;
            randomise_gru (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 8 && out_size == 8)
        {
            ModelT<double, 8, 8, GRULayerT<double, 8, 8>> model;
            randomise_gru (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 16 && out_size == 16)
        {
            ModelT<double, 16, 16, GRULayerT<double, 16, 16>> model;
            randomise_gru (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 1 && out_size == 24)
        {
            ModelT<double, 1, 24, GRULayerT<double, 1, 24>> model;
            randomise_gru (model.get<0>());
            duration = run_layer(model);
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }
    else if(layer_type == "lstm")
    {
        if(in_size == 4 && out_size == 4)
        {
            ModelT<double, 4, 4, LSTMLayerT<double, 4, 4>> model;
            randomise_lstm (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 8 && out_size == 8)
        {
            ModelT<double, 8, 8, LSTMLayerT<double, 8, 8>> model;
            randomise_lstm (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 16 && out_size == 16)
        {
            ModelT<double, 16, 16, LSTMLayerT<double, 16, 16>> model;
            randomise_lstm (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 1 && out_size == 24)
        {
            ModelT<double, 1, 24, LSTMLayerT<double, 1, 24>> model;
            randomise_lstm (model.get<0>());
            duration = run_layer(model);
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }
    else if(layer_type == "tanh")
    {
        if(in_size == 4 && out_size == 4)
        {
            ModelT<double, 4, 4, TanhActivationT<double, 4>> model;
            duration = run_layer(model);
        }
        else if(in_size == 8 && out_size == 8)
        {
            ModelT<double, 8, 8, TanhActivationT<double, 8>> model;
            duration = run_layer(model);
        }
        else if(in_size == 16 && out_size == 16)
        {
            ModelT<double, 16, 16, TanhActivationT<double, 16>> model;
            duration = run_layer(model);
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }
    else if(layer_type == "fast_tanh")
    {
        if(in_size == 4 && out_size == 4)
        {
            ModelT<double, 4, 4, FastTanhT<double, 4>> model;
            duration = run_layer(model);
        }
        else if(in_size == 8 && out_size == 8)
        {
            ModelT<double, 8, 8, FastTanhT<double, 8>> model;
            duration = run_layer(model);
        }
        else if(in_size == 16 && out_size == 16)
        {
            ModelT<double, 16, 16, FastTanhT<double, 16>> model;
            duration = run_layer(model);
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }
    else if(layer_type == "relu")
    {
        if(in_size == 4 && out_size == 4)
        {
            ModelT<double, 4, 4, ReLuActivationT<double, 4>> model;
            duration = run_layer(model);
        }
        else if(in_size == 8 && out_size == 8)
        {
            ModelT<double, 8, 8, ReLuActivationT<double, 8>> model;
            duration = run_layer(model);
        }
        else if(in_size == 16 && out_size == 16)
        {
            ModelT<double, 16, 16, ReLuActivationT<double, 16>> model;
            duration = run_layer(model);
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }
    else if(layer_type == "sigmoid")
    {
        if(in_size == 4 && out_size == 4)
        {
            ModelT<double, 4, 4, SigmoidActivationT<double, 4>> model;
            duration = run_layer(model);
        }
        else if(in_size == 8 && out_size == 8)
        {
            ModelT<double, 8, 8, SigmoidActivationT<double, 8>> model;
            duration = run_layer(model);
        }
        else if(in_size == 16 && out_size == 16)
        {
            ModelT<double, 16, 16, SigmoidActivationT<double, 16>> model;
            duration = run_layer(model);
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }
    else if(layer_type == "softmax")
    {
        if(in_size == 4 && out_size == 4)
        {
            ModelT<double, 4, 4, SoftmaxActivationT<double, 4>> model;
            duration = run_layer(model);
        }
        else if(in_size == 8 && out_size == 8)
        {
            ModelT<double, 8, 8, SoftmaxActivationT<double, 8>> model;
            duration = run_layer(model);
        }
        else if(in_size == 16 && out_size == 16)
        {
            ModelT<double, 16, 16, SoftmaxActivationT<double, 16>> model;
            duration = run_layer(model);
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }

    return duration;
}

#endif // MODELT_AVAILABLE
