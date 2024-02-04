#pragma once

#include <cmath>

namespace RTNEURAL_NAMESPACE
{
struct DefaultMathsProvider
{
    template <typename T>
    static void tanh(const T& x, T& y)
    {
        using std::tanh;
        using xsimd::tanh;
        y = tanh(x);
    }

    template <typename T>
    static void sigmoid(const T& x, T& y)
    {
        using std::exp;
        using xsimd::exp;
        y = (T)1 / ((T)1 + exp(-x));
    }

    template <typename T>
    static void exp(const T& x, T& y)
    {
        using std::exp;
        using xsimd::exp;
        y = exp(x);
    }
};
}
