#pragma once

#include <RTNeural.h>

void modelt_test()
{
    using TestType = float;

    // RTNeural::ModelT<float,
    //     RTNeural::Dense<TestType>,
    //     RTNeural::GRULayer<TestType>,
    //     RTNeural::LSTMLayer<TestType>,
    //     RTNeural::Conv1D<TestType>,
    //     RTNeural::TanhActivation<TestType>
    // > model ({2, 2, 2, 2, 2, 2});

    // std::initializer_list<size_t> ll { 2, 2 };
    // RTNeural::Dense<TestType> dd ({ ll.begin(), ll.end() });
}
