#pragma once

#include <cmath>

namespace RTNEURAL_NAMESPACE
{
struct DefaultMathsProvider
{
    template <typename T>
    static void tanh(const T& x, T& y)
    {
        y = std::tanh(x);
    }

    template <typename T>
    static void sigmoid(const T& x, T& y)
    {
        y = (T)1 / ((T)1 + std::exp(-x));
    }

    template <typename T>
    static void exp(const T& x, T& y)
    {
        y =  std::exp(x);
    }
};
}
