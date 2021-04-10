#ifndef ACTIVATION_H_INCLUDED
#define ACTIVATION_H_INCLUDED

#include "../Layer.h"
#include <functional>

namespace RTNeural
{

template <typename T>
class Activation : public Layer<T>
{
public:
    Activation(size_t size, std::function<T(T)> func)
        : Layer<T>(size, size)
        , func(func)
    {
    }

    inline void forward(const T* input, T* out) override
    {
        for(size_t i = 0; i < Layer<T>::out_size; ++i)
            out[i] = func(input[i]);
    }

private:
    const std::function<T(T)> func;
};

} // namespace RTNeural

#if defined(USE_EIGEN)
#include "activation_eigen.h"

#elif defined(USE_XSIMD)
#include "activation_xsimd.h"

#elif defined(USE_ACCELERATE)
#include "activation_accelerate.h"

#else
#include "../common.h"
#include <cmath>

namespace RTNeural
{

template <typename T>
class TanhActivation : public Activation<T>
{
public:
    TanhActivation(size_t size)
        : Activation<T>(size, [](T x) { return std::tanh(x); })
    {
    }

    TanhActivation(std::initializer_list<size_t> sizes)
        : TanhActivation(*sizes.begin())
    {
    }
};

template <typename T>
class ReLuActivation : public Activation<T>
{
public:
    ReLuActivation(size_t size)
        : Activation<T>(size, [](T x) { return std::max((T)0, x); })
    {
    }

    ReLuActivation(std::initializer_list<size_t> sizes)
        : ReluActivation(*sizes.begin())
    {
    }
};

template <typename T>
class SigmoidActivation : public Activation<T>
{
public:
    SigmoidActivation(size_t size)
        : Activation<T>(size, [](T x) { return sigmoid(x); })
    {
    }

    SigmoidActivation(std::initializer_list<size_t> sizes)
        : SigmoidActivation(*sizes.begin())
    {
    }
};

} // namespace RTNeural

#endif // USE_EIGEN

#endif // ACTIVATION_H_INCLUDED
