#ifndef ACTIVATIONXSIMD_H_INCLUDED
#define ACTIVATIONXSIMD_H_INCLUDED

#include "../common.h"

namespace RTNeural
{

template <typename T>
class TanhActivation : public Activation<T>
{
public:
    TanhActivation(int size)
        : Activation<T>(size, {}, "tanh")
    {
    }

    TanhActivation(std::initializer_list<int> sizes)
        : TanhActivation(*sizes.begin())
    {
    }

    inline void forward(const T* input, T* out) override
    {
        tanh(input, out, Layer<T>::in_size);
    }
};

template <typename T, int size>
class TanhActivationT
{
    using v_type = xsimd::simd_type<T>;
    static constexpr auto v_size = (int) v_type::size;
    static constexpr auto v_io_size = ceil_div(size, v_size);

public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    TanhActivationT()
    {
        for(int i = 0; i < v_io_size; ++i)
            outs[i] = v_type((T)0);
    }

    std::string getName() const noexcept { return "tanh"; }
    constexpr bool isActivation() const noexcept { return true; }
    void reset() { }

    inline void forward(const v_type (&ins)[v_io_size])
    {
        for(int i = 0; i < v_io_size; ++i)
            outs[i] = xsimd::tanh(ins[i]);
    }

    v_type outs[v_io_size];
};

template <typename T>
class ReLuActivation : public Activation<T>
{
public:
    ReLuActivation(int size)
        : Activation<T>(size, {}, "relu")
    {
        zeros.resize(size, (T)0);
    }

    ReLuActivation(std::initializer_list<int> sizes)
        : ReLuActivation(*sizes.begin())
    {
    }

    inline void forward(const T* input, T* out) override
    {
        xsimd::transform(
            input, &input[Layer<T>::in_size], zeros.begin(), out,
            [](auto const& a, auto const& b) { return xsimd::max(a, b); });
    }

    std::vector<T, XSIMD_DEFAULT_ALLOCATOR(T)> zeros;
};

template <typename T, int size>
class ReLuActivationT
{
    using v_type = xsimd::simd_type<T>;
    static constexpr auto v_size = (int) v_type::size;
    static constexpr auto v_io_size = ceil_div(size, v_size);

public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    ReLuActivationT()
    {
        for(int i = 0; i < v_io_size; ++i)
            outs[i] = v_type((T)0);
    }

    std::string getName() const noexcept { return "relu"; }
    constexpr bool isActivation() const noexcept { return true; }
    void reset() { }

    inline void forward(const v_type (&ins)[v_io_size])
    {
        for(int i = 0; i < v_io_size; ++i)
            outs[i] = xsimd::max(ins[i], v_type((T)0));
    }

    v_type outs[v_io_size];
};

template <typename T>
class SigmoidActivation : public Activation<T>
{
public:
    SigmoidActivation(int size)
        : Activation<T>(size, {}, "sigmoid")
    {
    }

    SigmoidActivation(std::initializer_list<int> sizes)
        : SigmoidActivation(*sizes.begin())
    {
    }

    inline void forward(const T* input, T* out) override
    {
        sigmoid(input, out, Layer<T>::in_size);
    }
};

template <typename T, int size>
class SigmoidActivationT
{
    using v_type = xsimd::simd_type<T>;
    static constexpr auto v_size = (int) v_type::size;
    static constexpr auto v_io_size = ceil_div(size, v_size);

public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    SigmoidActivationT()
    {
        for(int i = 0; i < v_io_size; ++i)
            outs[i] = v_type((T)0);
    }

    std::string getName() const noexcept { return "sigmoid"; }
    constexpr bool isActivation() const noexcept { return true; }
    void reset() { }

    inline void forward(const v_type (&ins)[v_io_size])
    {
        for(int i = 0; i < v_io_size; ++i)
            outs[i] = (T)1.0 / ((T)1.0 + xsimd::exp(-ins[i]));
    }

    v_type outs[v_io_size];
};

template <typename T>
class SoftmaxActivation : public Activation<T>
{
public:
    SoftmaxActivation(int size)
        : Activation<T>(size, {}, "softmax")
    {
    }

    SoftmaxActivation(std::initializer_list<int> sizes)
        : SoftmaxActivation(*sizes.begin())
    {
    }

    inline void forward(const T* input, T* out) override
    {
        softmax(input, out, Layer<T>::in_size);
    }
};

template <typename T, int size>
class SoftmaxActivationT
{
    using v_type = xsimd::simd_type<T>;
    static constexpr auto v_size = (int) v_type::size;
    static constexpr auto v_io_size = ceil_div(size, v_size);

public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    SoftmaxActivationT()
    {
        for(int i = 0; i < v_io_size; ++i)
            outs[i] = v_type((T)0);
    }

    std::string getName() const noexcept { return "softmax"; }
    constexpr bool isActivation() const noexcept { return true; }
    void reset() { }

    inline void forward(const v_type (&ins)[v_io_size])
    {
        auto exp_sum = (T)0.0;
        for(int i = 0; i < v_io_size; ++i)
        {
            outs[i] = xsimd::exp(ins[i]);
            exp_sum += xsimd::hadd(outs[i]);
        }

        auto v_exp_sum = (v_type)exp_sum;
        for(int i = 0; i < v_io_size; ++i)
            outs[i] = outs[i] / v_exp_sum;
    }

    v_type outs[v_io_size];
};

} // namespace RTNeural

#endif // ACTIVATIONXSIMD_H_INCLUDED
