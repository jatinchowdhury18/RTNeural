#ifndef ACTIVATIONACCELERATE_H_INCLUDED
#define ACTIVATIONACCELERATE_H_INCLUDED

#include "../common.h"

namespace RTNeural
{

/** Dynamic implementation of a tanh activation layer. */
template <typename T>
class TanhActivation : public Activation<T>
{
public:
    /** Constructs a tanh activation layer for a given size. */
    TanhActivation(int size)
        : Activation<T>(size, {}, "tanh")
    {
    }

    TanhActivation(std::initializer_list<int> sizes)
        : TanhActivation(*sizes.begin())
    {
    }

    /** Performs forward propagation for tanh activation. */
    inline void forward(const T* input, T* out) noexcept override
    {
        forward_internal(input, out);
    }

private:
    template <typename FloatType = T>
    inline typename std::enable_if<std::is_same<FloatType, float>::value>::type
    forward_internal(const float* input, float* out) noexcept
    {
        const auto dim_int = static_cast<int>(Layer<T>::in_size);
        vvtanhf(out, input, &dim_int);
    }

    template <typename FloatType = T>
    inline typename std::enable_if<std::is_same<FloatType, double>::value>::type
    forward_internal(const double* input, double* out) noexcept
    {
        const auto dim_int = static_cast<int>(Layer<T>::in_size);
        vvtanh(out, input, &dim_int);
    }
};

/** Dynamic implementation of a ReLU activation layer. */
template <typename T>
class ReLuActivation : public Activation<T>
{
public:
    /** Constructs a ReLU activation layer for a given size. */
    ReLuActivation(int size)
        : Activation<T>(size, {}, "relu")
    {
        zeros.resize(size, (T)0);
    }

    ReLuActivation(std::initializer_list<int> sizes)
        : ReLuActivation(*sizes.begin())
    {
    }

    /** Performs forward propagation for ReLU activation. */
    inline void forward(const T* input, T* out) noexcept override
    {
        forward_internal(input, out);
    }

private:
    template <typename FloatType = T>
    inline typename std::enable_if<std::is_same<FloatType, float>::value>::type
    forward_internal(const float* input, float* out) noexcept
    {
        vDSP_vmax(input, 1, zeros.data(), 1, out, 1, Layer<T>::in_size);
    }

    template <typename FloatType = T>
    inline typename std::enable_if<std::is_same<FloatType, double>::value>::type
    forward_internal(const double* input, double* out) noexcept
    {
        vDSP_vmaxD(input, 1, zeros.data(), 1, out, 1, Layer<T>::in_size);
    }

    std::vector<T> zeros;
};

/** Dynamic implementation of a sigmoid activation layer. */
template <typename T>
class SigmoidActivation : public Activation<T>
{
public:
    /** Constructs a sigmoid activation layer for a given size. */
    SigmoidActivation(int size)
        : Activation<T>(size, {}, "sigmoid")
    {
    }

    SigmoidActivation(std::initializer_list<int> sizes)
        : SigmoidActivation(*sizes.begin())
    {
    }

    /** Performs forward propagation for sigmoid activation. */
    inline void forward(const T* input, T* out) noexcept override
    {
        sigmoid(input, out, Layer<T>::in_size);
    }
};

/** Dynamic implementation of a softmax activation layer. */
template <typename T>
class SoftmaxActivation : public Activation<T>
{
public:
    /** Constructs a softmax activation layer for a given size. */
    SoftmaxActivation(int size)
        : Activation<T>(size, {}, "softmax")
    {
    }

    SoftmaxActivation(std::initializer_list<int> sizes)
        : SoftmaxActivation(*sizes.begin())
    {
    }

    /** Performs forward propagation for softmax activation. */
    inline void forward(const T* input, T* out) noexcept override
    {
        softmax(input, out, Layer<T>::in_size);
    }
};

} // namespace RTNeural

#endif // ACTIVATIONACCELERATE_H_INCLUDED
