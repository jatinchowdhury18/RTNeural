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
    Activation(int size, std::function<T(T)> func, std::string name)
        : Layer<T>(size, size)
        , name(name)
        , func(func)
    {
    }

    std::string getName() const noexcept override { return name; }

    inline void forward(const T* input, T* out) override
    {
        for(int i = 0; i < Layer<T>::out_size; ++i)
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
    TanhActivation(int size)
        : Activation<T>(
            size, [](T x) { return std::tanh(x); }, "tanh")
    {
    }

    TanhActivation(std::initializer_list<int> sizes)
        : TanhActivation(*sizes.begin())
    {
    }

    inline void forward(const T* input, T* out) override
    {
        for(int i = 0; i < Layer<T>::out_size; ++i)
            out[i] = std::tanh(input[i]);
    }
};

template <typename T, int size>
class TanhActivationT
{
public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    TanhActivationT() = default;

    std::string getName() const noexcept { return "tanh"; }
    constexpr bool isActivation() const noexcept { return true; }
    void reset() { }

    inline void forward(const T (&ins)[size])
    {
        for(int i = 0; i < size; ++i)
            outs[i] = std::tanh(ins[i]);
    }

    T outs alignas(16)[size];
};

template <typename T>
class ReLuActivation : public Activation<T>
{
public:
    ReLuActivation(int size)
        : Activation<T>(
            size, [](T x) { return std::max((T)0, x); }, "relu")
    {
    }

    ReLuActivation(std::initializer_list<int> sizes)
        : ReLuActivation(*sizes.begin())
    {
    }
};

template <typename T, int size>
class ReLuActivationT
{
public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    ReLuActivationT() = default;

    std::string getName() const noexcept { return "relu"; }
    constexpr bool isActivation() const noexcept { return true; }
    void reset() { }

    inline void forward(const T (&ins)[size])
    {
        for(int i = 0; i < size; ++i)
            outs[i] = std::max((T)0, ins[i]);
    }

    T outs alignas(16)[size];
};

template <typename T>
class SigmoidActivation : public Activation<T>
{
public:
    SigmoidActivation(int size)
        : Activation<T>(
            size, [](T x) { return sigmoid(x); }, "sigmoid")
    {
    }

    SigmoidActivation(std::initializer_list<int> sizes)
        : SigmoidActivation(*sizes.begin())
    {
    }
};

template <typename T, int size>
class SigmoidActivationT
{
public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    SigmoidActivationT() = default;

    std::string getName() const noexcept { return "sigmoid"; }
    constexpr bool isActivation() const noexcept { return true; }
    void reset() { }

    inline void forward(const T (&ins)[size])
    {
        for(int i = 0; i < size; ++i)
            outs[i] = sigmoid(ins[i]);
    }

    T outs alignas(16)[size];
};

template <typename T>
class SoftmaxActivation : public Activation<T>
{
public:
    SoftmaxActivation(int size)
        : Activation<T>(
            size, [](T x) { return (T)0; }, "softmax")
    {
    }

    SoftmaxActivation(std::initializer_list<int> sizes)
        : SoftmaxActivation(*sizes.begin())
    {
    }

    inline void forward(const T* input, T* out) override
    {
        softmax(input, out, Layer<T>::out_size);
    }
};

template <typename T, int size>
class SoftmaxActivationT
{
public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    SoftmaxActivationT() = default;

    std::string getName() const noexcept { return "softmax"; }
    constexpr bool isActivation() const noexcept { return true; }
    void reset() { }

    inline void forward(const T (&ins)[size])
    {
        T exp_sum = 0;
        for(int i = 0; i < size; ++i)
        {
            outs[i] = std::exp(ins[i]);
            exp_sum += outs[i];
        }

        for(int i = 0; i < size; ++i)
        {
            outs[i] /= exp_sum;
        }
    }

    T outs alignas(16)[size];
};

} // namespace RTNeural

#endif // USE_EIGEN

#endif // ACTIVATION_H_INCLUDED
