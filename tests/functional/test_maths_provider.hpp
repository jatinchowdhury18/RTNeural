#pragma once

#include <RTNeural/RTNeural.h>

#if RTNEURAL_USE_XSIMD
struct TestMathsProvider
{
    template <typename T>
    static T tanh(T x) { using std::tanh; using xsimd::tanh; return tanh(x); }
    template <typename T>
    static T sigmoid(T x) { using std::exp; using xsimd::exp; return (T)1 / ((T)1 + exp(-x)); }
    template <typename T>
    static T exp(T x) { using std::exp; using xsimd::exp; return exp(x); }
};
#elif RTNEURAL_USE_EIGEN
struct TestMathsProvider
{
    template <typename Matrix>
    static auto tanh(const Matrix& x) { return x.array().tanh(); }
    template <typename Matrix>
    static auto sigmoid(const Matrix& x) { using T = typename Matrix::Scalar; return (T)1 / (((T)-1 * x.array()).array().exp() + (T)1); }
    template <typename Matrix>
    static auto exp(const Matrix& x) { return x.array().exp(); }
};
#else
struct TestMathsProvider
{
    template <typename T>
    static T tanh(T x) { return std::tanh(x); }
    template <typename T>
    static T sigmoid(T x) { return (T)1 / ((T)1 + std::exp(-x)); }
    template <typename T>
    static T exp(T x) { return std::exp(x); }
};
#endif
