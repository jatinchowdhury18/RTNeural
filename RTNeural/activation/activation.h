#ifndef ACTIVATION_H_INCLUDED
#define ACTIVATION_H_INCLUDED

#include "../Layer.h"
#include <functional>

namespace RTNeural
{

/** Base class for activation layers. */
template <typename T>
class Activation : public Layer<T>
{
public:
    /** Constructs an activation layers for a given size and function. */
    Activation(int size, std::function<T(T)> func, std::string name)
        : Layer<T>(size, size)
        , name(name)
        , func(func)
    {
    }

    /** Returns the name of this layer. */
    std::string getName() const noexcept override { return name; }

    /** Implements the forward propagation step for this layer. */
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

#if RTNEURAL_USE_EIGEN
#include "activation_eigen.h"

#elif RTNEURAL_USE_XSIMD
#include "activation_xsimd.h"

#elif RTNEURAL_USE_ACCELERATE
#include "activation_accelerate.h"

#else
#include "../common.h"
#include <cmath>

namespace RTNeural
{

/** Dynamic implementation of a tanh activation layer. */
template <typename T>
class TanhActivation final : public Activation<T>
{
public:
    /** Constructs a tanh activation layer for a given size. */
    TanhActivation(int size)
        : Activation<T>(
            size, [](T x) { return std::tanh(x); }, "tanh")
    {
    }

    TanhActivation(std::initializer_list<int> sizes)
        : TanhActivation(*sizes.begin())
    {
    }

    /** Performs forward propagation for tanh activation. */
    inline void forward(const T* input, T* out) override
    {
        for(int i = 0; i < Layer<T>::out_size; ++i)
            out[i] = std::tanh(input[i]);
    }
};

/** Static implementation of a tanh activation layer. */
template <typename T, int size>
class TanhActivationT
{
public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    TanhActivationT() = default;

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "tanh"; }

    /** Returns true since this layer is an activation layer. */
    constexpr bool isActivation() const noexcept { return true; }

    void reset() { }

    /** Performs forward propagation for tanh activation. */
    inline void forward(const T (&ins)[size])
    {
        for(int i = 0; i < size; ++i)
            outs[i] = std::tanh(ins[i]);
    }

    T outs alignas(RTNEURAL_DEFAULT_ALIGNMENT)[size];
};

/** Dynamic implementation of an approximate tanh activation layer. */
template <typename T>
class FastTanh final : public Activation<T>
{
public:
    /** Constructs a tanh activation layer for a given size. */
    FastTanh(int size)
        : Activation<T>(
            size, [](T x) { return tanh_approx(x); }, "tanh")
    {
    }

    FastTanh(std::initializer_list<int> sizes)
        : FastTanh(*sizes.begin())
    {
    }

    /** Performs forward propagation for tanh activation. */
    inline void forward(const T* input, T* out) override
    {
        for(int i = 0; i < Layer<T>::out_size; ++i)
            out[i] = tanh_approx(input[i]);
    }
};

/** Static implementation of an approximate tanh activation layer. */
template <typename T, int size>
class FastTanhT
{
public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    FastTanhT() = default;

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "tanh"; }

    /** Returns true since this layer is an activation layer. */
    constexpr bool isActivation() const noexcept { return true; }

    void reset() { }

    /** Performs forward propagation for tanh activation. */
    inline void forward(const T (&ins)[size])
    {
        for(int i = 0; i < size; ++i)
            outs[i] = tanh_approx(ins[i]);
    }

    T outs alignas(RTNEURAL_DEFAULT_ALIGNMENT)[size];
};

/** Dynamic implementation of a ReLU activation layer. */
template <typename T>
class ReLuActivation final : public Activation<T>
{
public:
    /** Constructs a ReLU activation layer for a given size. */
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

/** Static implementation of a ReLU activation layer. */
template <typename T, int size>
class ReLuActivationT
{
public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    ReLuActivationT() = default;

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "relu"; }

    /** Returns true since this layer is an activation layer. */
    constexpr bool isActivation() const noexcept { return true; }

    void reset() { }

    /** Performs forward propagation for ReLU activation. */
    inline void forward(const T (&ins)[size])
    {
        for(int i = 0; i < size; ++i)
            outs[i] = std::max((T)0, ins[i]);
    }

    T outs alignas(RTNEURAL_DEFAULT_ALIGNMENT)[size];
};

/** Dynamic implementation of a sigmoid activation layer. */
template <typename T>
class SigmoidActivation final : public Activation<T>
{
public:
    /** Constructs a sigmoid activation layer for a given size. */
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

/** Static implementation of a sigmoid activation layer. */
template <typename T, int size>
class SigmoidActivationT
{
public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    SigmoidActivationT() = default;

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "sigmoid"; }

    /** Returns true since this layer is an activation layer. */
    constexpr bool isActivation() const noexcept { return true; }

    void reset() { }

    /** Performs forward propagation for sigmoid activation. */
    inline void forward(const T (&ins)[size])
    {
        for(int i = 0; i < size; ++i)
            outs[i] = sigmoid(ins[i]);
    }

    T outs alignas(RTNEURAL_DEFAULT_ALIGNMENT)[size];
};

/** Dynamic implementation of a softmax activation layer. */
template <typename T>
class SoftmaxActivation final : public Activation<T>
{
public:
    /** Constructs a softmax activation layer for a given size. */
    SoftmaxActivation(int size)
        : Activation<T>(
            size, [](T x) { return (T)0; }, "softmax")
    {
    }

    SoftmaxActivation(std::initializer_list<int> sizes)
        : SoftmaxActivation(*sizes.begin())
    {
    }

    /** Performs forward propagation for softmax activation. */
    inline void forward(const T* input, T* out) override
    {
        softmax(input, out, Layer<T>::out_size);
    }
};

/** Static implementation of a softmax activation layer. */
template <typename T, int size>
class SoftmaxActivationT
{
public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    SoftmaxActivationT() = default;

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "softmax"; }

    /** Returns true since this layer is an activation layer. */
    constexpr bool isActivation() const noexcept { return true; }

    void reset() { }

    /** Performs forward propagation for softmax activation. */
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

    T outs alignas(RTNEURAL_DEFAULT_ALIGNMENT)[size];
};

} // namespace RTNeural

#endif // RTNEURAL_USE_EIGEN

#endif // ACTIVATION_H_INCLUDED
