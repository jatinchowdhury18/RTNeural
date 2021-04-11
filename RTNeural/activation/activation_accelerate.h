#ifndef ACTIVATIONACCELERATE_H_INCLUDED
#define ACTIVATIONACCELERATE_H_INCLUDED

#include "../common.h"

namespace RTNeural
{

template <typename T>
class TanhActivation : public Activation<T>
{
public:
    TanhActivation(size_t size)
        : Activation<T>(size, {}, "tanh")
    {
    }

    TanhActivation(std::initializer_list<size_t> sizes)
        : TanhActivation(*sizes.begin())
    {
    }

    inline void forward(const T* input, T* out) override
    {
        forward_internal(input, out);
    }

private:
    template <typename FloatType = T>
    inline typename std::enable_if<std::is_same<FloatType, float>::value>::type
    forward_internal(const float* input, float* out)
    {
        const auto dim_int = static_cast<int>(Layer<T>::in_size);
        vvtanhf(out, input, &dim_int);
    }

    template <typename FloatType = T>
    inline typename std::enable_if<std::is_same<FloatType, double>::value>::type
    forward_internal(const double* input, double* out)
    {
        const auto dim_int = static_cast<int>(Layer<T>::in_size);
        vvtanh(out, input, &dim_int);
    }
};

template <typename T>
class ReLuActivation : public Activation<T>
{
public:
    ReLuActivation(size_t size)
        : Activation<T>(size, {}, "relu")
    {
        zeros.resize(size, (T)0);
    }

    ReLuActivation(std::initializer_list<size_t> sizes)
        : ReLuActivation(*sizes.begin())
    {
    }

    inline void forward(const T* input, T* out) override
    {
        forward_internal(input, out);
    }

private:
    template <typename FloatType = T>
    inline typename std::enable_if<std::is_same<FloatType, float>::value>::type
    forward_internal(const float* input, float* out)
    {
        vDSP_vmax(input, 1, zeros.data(), 1, out, 1, Layer<T>::in_size);
    }

    template <typename FloatType = T>
    inline typename std::enable_if<std::is_same<FloatType, double>::value>::type
    forward_internal(const double* input, double* out)
    {
        vDSP_vmaxD(input, 1, zeros.data(), 1, out, 1, Layer<T>::in_size);
    }

    std::vector<T> zeros;
};

template <typename T>
class SigmoidActivation : public Activation<T>
{
public:
    SigmoidActivation(size_t size)
        : Activation<T>(size, {}, "sigmoid")
    {
    }

    SigmoidActivation(std::initializer_list<size_t> sizes)
        : SigmoidActivation(*sizes.begin())
    {
    }

    inline void forward(const T* input, T* out) override
    {
        sigmoid(input, out, Layer<T>::in_size);
    }
};

} // namespace RTNeural

#endif // ACTIVATIONACCELERATE_H_INCLUDED
