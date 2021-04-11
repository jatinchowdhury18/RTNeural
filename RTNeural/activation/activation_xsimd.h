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
        : Activation<T>(size, {}, "tanh")
    {
    }

    TanhActivation(std::initializer_list<size_t> sizes)
        : TanhActivation(*sizes.begin())
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

#endif // ACTIVATIONXSIMD_H_INCLUDED
