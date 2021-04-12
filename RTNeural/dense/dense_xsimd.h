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
    static size_t ceil_div(size_t n, size_t d)
    {
        return (n + d - 1) / d;
    }

    Dense(size_t in_size, size_t out_size)
        : Layer<T>(in_size, out_size)
    {
        std::cout << "Dense - XSIMD NEW" << std::endl;

        prod.resize(in_size, (T)0);
        weights = std::vector<vec_type>(out_size, vec_type(in_size, (T)0));

        bias.resize(out_size, (T)0);
        sums.resize(out_size, (T)0);
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
    }

    inline void forward(const T* input, T* out) override
    {
        for(size_t l = 0; l < Layer<T>::out_size; ++l)
        {
            xsimd::transform(input, &input[Layer<T>::in_size], weights[l].data(), prod.data(),
                [](auto const& a, auto const& b) { return a * b; });

            auto sum = xsimd::reduce(prod.data(), &prod[Layer<T>::in_size], (T)0);
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
    using vec_type = std::vector<T, XSIMD_DEFAULT_ALLOCATOR(T)>;

    vec_type bias;
    std::vector<vec_type> weights;
    vec_type prod;
    vec_type sums;
};

} // namespace RTNeural

#endif // DENSEXSIMD_H_INCLUDED
