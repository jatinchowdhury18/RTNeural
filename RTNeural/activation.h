#ifndef ACTIVATION_H_INCLUDED
#define ACTIVATION_H_INCLUDED

#include "Layer.h"
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

    virtual ~Activation() { }

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
#include "common.h"

namespace RTNeural
{

template <typename T>
class TanhActivation : public Activation<T>
{
public:
    TanhActivation(size_t size)
        : Activation<T>(size, {})
    {
        inVec.resize(size, 1);
        outVec.resize(size, 1);
    }

    inline void forward(const T* input, T* out) override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(
            input, Layer<T>::in_size, 1);
        outVec = inVec.array().tanh();

        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
};

template <typename T>
class ReLuActivation : public Activation<T>
{
public:
    ReLuActivation(size_t size)
        : Activation<T>(size, {})
    {
        inVec.resize(size, 1);
        outVec.resize(size, 1);
    }

    inline void forward(const T* input, T* out) override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(
            input, Layer<T>::in_size, 1);
        outVec = inVec.array().max((T)0);

        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
};

template <typename T>
class SigmoidActivation : public Activation<T>
{
public:
    SigmoidActivation(size_t size)
        : Activation<T>(size, {})
    {
        inVec.resize(size, 1);
        outVec.resize(size, 1);
    }

    inline void forward(const T* input, T* out) override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(
            input, Layer<T>::in_size, 1);
        outVec = inVec.array();
        sigmoid(outVec);

        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
};

} // namespace RTNeural

#elif defined(USE_XSIMD)
#include "common.h"

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

#else
#include "common.h"
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
};

template <typename T>
class ReLuActivation : public Activation<T>
{
public:
    ReLuActivation(size_t size)
        : Activation<T>(size, [](T x) { return std::max((T)0, x); })
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
};

} // namespace RTNeural

#endif // USE_EIGEN

#endif // ACTIVATION_H_INCLUDED
