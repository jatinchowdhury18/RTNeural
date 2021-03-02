#ifndef ACTIVATIONXSIMD_H_INCLUDED
#define ACTIVATIONXSIMD_H_INCLUDED

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
        tanh(input, out, Layer<T>::in_size);
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
        xsimd::transform(
            input, &input[Layer<T>::in_size], zeros.begin(), out,
            [](auto const& a, auto const& b) { return xsimd::max(a, b); });
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

#endif // ACTIVATIONXSIMD_H_INCLUDED
