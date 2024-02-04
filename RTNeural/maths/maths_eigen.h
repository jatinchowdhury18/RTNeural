#pragma once

#include <cmath>

namespace RTNEURAL_NAMESPACE
{
struct DefaultMathsProvider
{
    template <typename MatrixX, typename MatrixY>
    static auto tanh(const MatrixX& x, MatrixY& y)
    {
        y =  x.array().tanh();
    }

    template <typename MatrixX, typename MatrixY>
    static auto sigmoid(const MatrixX& x, MatrixY& y)
    {
        using T = typename MatrixX::Scalar;
        y = (T)1 / (((T)-1 * x.array()).array().exp() + (T)1);
    }

    template <typename MatrixX, typename MatrixY>
    static auto exp(const MatrixX& x, MatrixY& y)
    {
        y = x.array().exp();
    }
};
}
