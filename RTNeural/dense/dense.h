#ifndef DENSE_H_INCLUDED
#define DENSE_H_INCLUDED

#include <algorithm>
#include <numeric>
#include <vector>

#if defined(USE_EIGEN)
#include "dense_eigen.h"
#elif defined(USE_XSIMD)
#include "dense_xsimd.h"
#elif defined(USE_ACCELERATE)
#include "dense_accelerate.h"
#else
#include "../Layer.h"

namespace RTNeural
{

template <typename T>
class Dense1
{
public:
    Dense1(size_t in_size)
        : in_size(in_size)
    {
        weights = new T[in_size];
    }

    ~Dense1() { delete[] weights; }

    inline T forward(const T* input)
    {
        return std::inner_product(weights, weights + in_size, input, (T)0) + bias;
    }

    void setWeights(const T* newWeights)
    {
        for(size_t i = 0; i < in_size; ++i)
            weights[i] = newWeights[i];
    }

    void setBias(T b) { bias = b; }

    T getWeight(size_t i) const noexcept { return weights[i]; }

    T getBias() const noexcept { return bias; }

private:
    const size_t in_size;
    T bias;

    T* weights;
};

template <typename T>
class Dense : public Layer<T>
{
public:
    Dense(size_t in_size, size_t out_size)
        : Layer<T>(in_size, out_size)
    {
        subLayers = new Dense1<T>*[out_size];
        for(size_t i = 0; i < out_size; ++i)
            subLayers[i] = new Dense1<T>(in_size);
    }

    Dense(std::initializer_list<size_t> sizes)
        : Dense(*sizes.begin(), *(sizes.begin() + 1))
    {
    }

    Dense(const Dense& other)
        : Dense(other.in_size, other.out_size)
    {
    }

    Dense& operator=(const Dense& other)
    {
        return *this = Dense(other);
    }

    virtual ~Dense()
    {
        for(size_t i = 0; i < Layer<T>::out_size; ++i)
            delete subLayers[i];

        delete[] subLayers;
    }

    std::string getName() const noexcept override { return "dense"; }

    inline void forward(const T* input, T* out) override
    {
        for(size_t i = 0; i < Layer<T>::out_size; ++i)
            out[i] = subLayers[i]->forward(input);
    }

    void setWeights(const std::vector<std::vector<T>>& newWeights)
    {
        for(size_t i = 0; i < Layer<T>::out_size; ++i)
            subLayers[i]->setWeights(newWeights[i].data());
    }

    void setWeights(T** newWeights)
    {
        for(size_t i = 0; i < Layer<T>::out_size; ++i)
            subLayers[i]->setWeights(newWeights[i]);
    }

    void setBias(T* b)
    {
        for(size_t i = 0; i < Layer<T>::out_size; ++i)
            subLayers[i]->setBias(b[i]);
    }

    T getWeight(size_t i, size_t k) const noexcept
    {
        return subLayers[i]->getWeight(k);
    }

    T getBias(size_t i) const noexcept { return subLayers[i]->getBias(); }

private:
    Dense1<T>** subLayers;
};

//====================================================
template <typename T, size_t in_sizet, size_t out_sizet>
class DenseT
{
    static constexpr auto weights_size = in_sizet * out_sizet;

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = out_sizet;

    DenseT()
    {
        for(size_t i = 0; i < weights_size; ++i)
            weights[i] = (T)0.0;

        for(size_t i = 0; i < out_size; ++i)
            bias[i] = (T)0.0;

        for(size_t i = 0; i < out_size; ++i)
            outs[i] = (T)0.0;
    }

    std::string getName() const noexcept { return "dense"; }
    constexpr bool isActivation() const noexcept { return false; }

    void reset() { }

    inline void forward(const T (&ins)[in_size])
    {
        for(size_t i = 0; i < out_size; ++i)
            outs[i] = std::inner_product (ins, ins + in_size, &weights[i * in_size], (T) 0) + bias[i];
    }

    void setWeights(const std::vector<std::vector<T>>& newWeights)
    {
        for(size_t i = 0; i < out_size; ++i)
        {
            for(size_t k = 0; k < in_size; ++k)
            {
                auto idx = i * in_size + k;
                weights[idx] = newWeights[i][k];
            }
        }
    }

    void setWeights(T** newWeights)
    {
        for(size_t i = 0; i < out_size; ++i)
        {
            for(size_t k = 0; k < in_size; ++k)
            {
                auto idx = i * in_size + k;
                weights[idx] = newWeights[i][k];
            }
        }
    }

    void setBias(T* b)
    {
        for(size_t i = 0; i < out_size; ++i)
            bias[i] = b[i];
    }

    T outs alignas(16)[out_size];

private:
    T bias[out_size];
    T weights[weights_size];
};

} // namespace RTNeural

#endif // USE_STL

#endif // DENSE_H_INCLUDED
