#ifndef DENSEACCELERATE_H_INCLUDED
#define DENSEACCELERATE_H_INCLUDED

#include "../Layer.h"
#include <Accelerate/Accelerate.h>

namespace RTNeural
{

/** Dynamic implementation of a fully-connected (dense) layer. */
template <typename T>
class Dense : public Layer<T>
{
public:
    /** Constructs a dense layer for a given input and output size. */
    Dense(int in_size, int out_size)
        : Layer<T>(in_size, out_size)
    {
        sums = new T[out_size];
        bias = new T[out_size];
        weights = new T*[out_size];
        for(int i = 0; i < out_size; ++i)
            weights[i] = new T[in_size];
    }

    Dense(std::initializer_list<int> sizes)
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
        for(int i = 0; i < Layer<T>::out_size; ++i)
            delete[] weights[i];
        delete[] weights;
    }

    /** Returns the name of this layer. */
    std::string getName() const noexcept override { return "dense"; }

    /** Performs forward propagation for this layer. */
    inline void forward(const T* input, T* out) noexcept override
    {
        forward_internal(input, out);
    }

    /** Sets the layer weights from a given vector. */
    void setWeights(const std::vector<std::vector<T>>& newWeights)
    {
        for(int i = 0; i < Layer<T>::out_size; ++i)
            for(int k = 0; k < Layer<T>::in_size; ++k)
                weights[i][k] = newWeights[i][k];
    }

    /** Sets the layer weights from a given array. */
    void setWeights(T** newWeights)
    {
        for(int i = 0; i < Layer<T>::out_size; ++i)
            for(int k = 0; k < Layer<T>::in_size; ++k)
                weights[i][k] = newWeights[i][k];
    }

    /** Sets the layer bias from a given array. */
    void setBias(const T* b)
    {
        for(int i = 0; i < Layer<T>::out_size; ++i)
            bias[i] = b[i];
    }

    /** Returns the weights value at the given indices. */
    T getWeight(int i, int k) const noexcept { return weights[i][k]; }

    /** Returns the bias value at the given index. */
    T getBias(int i) const noexcept { return bias[i]; }

private:
    template <typename FloatType = T>
    inline typename std::enable_if<std::is_same<FloatType, float>::value>::type
    forward_internal(const float* input, float* out) noexcept
    {
        for(int l = 0; l < Layer<T>::out_size; ++l)
            vDSP_dotpr(input, 1, weights[l], 1, &sums[l], Layer<T>::in_size);

        vDSP_vadd(sums, 1, bias, 1, out, 1, Layer<T>::out_size);
    }

    template <typename FloatType = T>
    inline typename std::enable_if<std::is_same<FloatType, double>::value>::type
    forward_internal(const double* input, double* out) noexcept
    {
        for(int l = 0; l < Layer<T>::out_size; ++l)
            vDSP_dotprD(input, 1, weights[l], 1, &sums[l], Layer<T>::in_size);

        vDSP_vaddD(sums, 1, bias, 1, out, 1, Layer<T>::out_size);
    }

    T* bias;
    T** weights;
    T* sums;
};

} // namespace RTNeural

#endif // DENSEACCELERATE_H_INCLUDED
