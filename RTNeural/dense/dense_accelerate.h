#ifndef DENSEACCELERATE_H_INCLUDED
#define DENSEACCELERATE_H_INCLUDED

#include "../Layer.h"
#include <Accelerate/Accelerate.h>

namespace RTNeural
{

template <typename T>
class Dense : public Layer<T>
{
public:
    Dense(size_t in_size, size_t out_size)
        : Layer<T>(in_size, out_size)
    {
        sums = new T[out_size];
        bias = new T[out_size];
        weights = new T*[out_size];
        for(size_t i = 0; i < out_size; ++i)
            weights[i] = new T[in_size];
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
        delete[] bias;
        delete[] sums;
        for(size_t i = 0; i < Layer<T>::out_size; ++i)
            delete[] weights[i];
        delete[] weights;
    }

    std::string getName() const noexcept override { return "dense"; }

    inline void forward(const T* input, T* out) override
    {
        forward_internal(input, out);
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
    template <typename FloatType = T>
    inline typename std::enable_if<std::is_same<FloatType, float>::value>::type
    forward_internal(const float* input, float* out)
    {
        for(size_t l = 0; l < Layer<T>::out_size; ++l)
            vDSP_dotpr(input, 1, weights[l], 1, &sums[l], Layer<T>::in_size);

        vDSP_vadd(sums, 1, bias, 1, out, 1, Layer<T>::out_size);
    }

    template <typename FloatType = T>
    inline typename std::enable_if<std::is_same<FloatType, double>::value>::type
    forward_internal(const double* input, double* out)
    {
        for(size_t l = 0; l < Layer<T>::out_size; ++l)
            vDSP_dotprD(input, 1, weights[l], 1, &sums[l], Layer<T>::in_size);

        vDSP_vaddD(sums, 1, bias, 1, out, 1, Layer<T>::out_size);
    }

    T* bias;
    T** weights;
    T* sums;
};

} // namespace RTNeural

#endif // DENSEACCELERATE_H_INCLUDED
