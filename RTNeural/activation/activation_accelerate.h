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
        : Activation<T>(size, {})
    {
    }

    inline void forward(const T* input, T* out) override
    {
        forward_internal(input, out);
    }

private:
    template<typename FloatType = T>
    inline typename std::enable_if <std::is_same<FloatType, float>::value>::type
    forward_internal(const float* input, float* out)
    {
        const auto dim_int = static_cast<int> (Layer<T>::in_size);
        vvtanhf(out, input, &dim_int);
    }

    template<typename FloatType = T>
    inline typename std::enable_if <std::is_same<FloatType, double>::value>::type
    forward_internal(const double* input, double* out)
    {
        const auto dim_int = static_cast<int> (Layer<T>::in_size);
        vvtanh(out, input, &dim_int);
    }
};

template <typename T>
class ReLuActivation : public Activation<T>
{
public:
    ReLuActivation(size_t size)
        : Activation<T>(size, {})
    {
        zeros.resize(size, (T)0);
    }

    inline void forward(const T* input, T* out) override
    {
        forward_internal(input, out);
    }

private:
    template<typename FloatType = T>
    inline typename std::enable_if <std::is_same<FloatType, float>::value>::type
    forward_internal(const float* input, float* out)
    {
        vDSP_vmax(input, 1, zeros.data(), 1, out, 1, Layer<T>::in_size);
    }

    template<typename FloatType = T>
    inline typename std::enable_if <std::is_same<FloatType, double>::value>::type
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
        : Activation<T>(size, {})
    {
    }

    inline void forward(const T* input, T* out) override
    {
        sigmoid(input, out, Layer<T>::in_size);
    }
};

} // namespace RTNeural

#endif // ACTIVATIONACCELERATE_H_INCLUDED
