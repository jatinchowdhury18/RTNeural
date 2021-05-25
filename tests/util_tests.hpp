#pragma once

#include <RTNeural.h>

using TestType = double;

template <typename T>
std::tuple<T> make_layer_tuple(std::initializer_list<int> args)
{
    return std::make_tuple(T(args));
}

void rule_of_three_test()
{
    std::cout << "\t Testing Dense..." << std::endl;
    auto dense = make_layer_tuple<RTNeural::Dense<TestType>>({ 2, 2 });

    std::cout << "\t Testing GRU..." << std::endl;
    auto gru = make_layer_tuple<RTNeural::GRULayer<TestType>>({ 2, 2 });

    std::cout << "\t Testing LSTM..." << std::endl;
    auto lstm = make_layer_tuple<RTNeural::LSTMLayer<TestType>>({ 2, 2 });

    std::cout << "\t Testing Conv1D..." << std::endl;
    auto conv1d = make_layer_tuple<RTNeural::Conv1D<TestType>>({ 2, 2, 1, 1 });
}

void util_test()
{
    std::cout << "Running Rule of Three Test:" << std::endl;
    rule_of_three_test();
}
