#ifndef DENSEXSIMD_H_INCLUDED
#define DENSEXSIMD_H_INCLUDED

#include "../Layer.h"
#include <xsimd/xsimd.hpp>

namespace RTNeural
{

template <typename T>
class Dense : public Layer<T>
{
public:
    Dense(size_t in_size, size_t out_size)
        : Layer<T>(in_size, out_size)
    {
        prod = new T[in_size];
        bias = new T[out_size];
        weights = new T*[out_size];
        for(size_t i = 0; i < out_size; ++i)
            weights[i] = new T[in_size];
    }

    virtual ~Dense()
    {
        delete[] bias;
        delete[] prod;
        for(size_t i = 0; i < Layer<T>::out_size; ++i)
            delete[] weights[i];
        delete[] weights;
    }

    inline void forward(const T* input, T* out) override
    {
        for(size_t l = 0; l < Layer<T>::out_size; ++l)
        {
            xsimd::transform(input, &input[Layer<T>::in_size], weights[l], prod,
                [](auto const& a, auto const& b) { return a * b; });

            auto sum = xsimd::reduce(prod, &prod[Layer<T>::in_size], (T)0);
            out[l] = sum + bias[l];
        }
    }

    void setWeights(const std::vector<std::vector<T>>& newWeights)
    {
        for(size_t i = 0; i < Layer<T>::out_size; ++i)
            for(size_t k = 0; k < Layer<T>::in_size; ++k)
                weights[i][k] = newWeights[i][k];
    }

    void setWeights(T** newWeights)
    {
        for(size_t i = 0; i < Layer<T>::out_size; ++i)
            for(size_t k = 0; k < Layer<T>::in_size; ++k)
                weights[i][k] = newWeights[i][k];
    }

    void setBias(T* b)
    {
        for(size_t i = 0; i < Layer<T>::out_size; ++i)
            bias[i] = b[i];
    }

    T getWeight(size_t i, size_t k) const noexcept { return weights[i][k]; }

    T getBias(size_t i) const noexcept { return bias[i]; }

private:
    T* bias;
    T** weights;
    T* prod;
};

} // namespace RTNeural

#endif // DENSEXSIMD_H_INCLUDED
