#pragma once

#include <cmath>

namespace RTNEURAL_NAMESPACE
{
struct DefaultMathsProvider
{
    template <typename Matrix>
    static auto tanh(const Matrix& x)
    {
        return x.array().tanh();
    }

    template <typename Matrix>
    static auto sigmoid(const Matrix& x)
    {
        using T = typename Matrix::Scalar;

        return ((x.array() / (T)2).array().tanh() + (T)1) / (T)2;
    }

    template <typename Matrix>
    static auto exp(const Matrix& x)
    {
        return x.array().exp();
    }
};
}
