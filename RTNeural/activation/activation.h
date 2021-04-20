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
    Activation(size_t size, std::function<T(T)> func, std::string name)
        : Layer<T>(size, size)
        , func(func)
        , name(name)
    {
    }

    std::string getName() const noexcept { return name; }

    inline void forward(const T* input, T* out) override
    {
        for(size_t i = 0; i < Layer<T>::out_size; ++i)
            out[i] = func(input[i]);
    }

private:
    const std::string name;
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
        : Activation<T>(
            size, [](T x) { return std::tanh(x); }, "tanh")
    {
    }

    TanhActivation(std::initializer_list<size_t> sizes)
        : TanhActivation(*sizes.begin())
    {
    }

    inline void forward(const T* input, T* out) override
    {
        for(size_t i = 0; i < Layer<T>::out_size; ++i)
            out[i] = std::tanh(input[i]);
    }
};

template <typename T>
class ReLuActivation : public Activation<T>
{
public:
    ReLuActivation(size_t size)
        : Activation<T>(
            size, [](T x) { return std::max((T)0, x); }, "relu")
    {
    }

    ReLuActivation(std::initializer_list<size_t> sizes)
        : ReLuActivation(*sizes.begin())
    {
    }
};

template <typename T>
class SigmoidActivation : public Activation<T>
{
public:
    SigmoidActivation(size_t size)
        : Activation<T>(
            size, [](T x) { return sigmoid(x); }, "sigmoid")
    {
    }

    SigmoidActivation(std::initializer_list<size_t> sizes)
        : SigmoidActivation(*sizes.begin())
    {
    }
};

template <typename T>
class SoftmaxActivation : public Activation<T>
{
public:
    SoftmaxActivation(size_t size)
        : Activation<T>(
            size, [](T x) { return (T)0; }, "softmax")
    {
    }

    SoftmaxActivation(std::initializer_list<size_t> sizes)
        : SoftmaxActivation(*sizes.begin())
    {
    }

    inline void forward(const T* input, T* out) override
    {
        softmax(input, out, Layer<T>::out_size);
    }
};

} // namespace RTNeural

#endif // USE_EIGEN

#endif // ACTIVATION_H_INCLUDED
