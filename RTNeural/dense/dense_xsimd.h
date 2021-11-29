#ifndef DENSEXSIMD_H_INCLUDED
#define DENSEXSIMD_H_INCLUDED

#include "../Layer.h"
#include <xsimd/xsimd.hpp>

namespace RTNeural
{

/**
 * Dynamic implementation of a fully-connected (dense) layer,
 * with no activation.
 */
template <typename T>
class Dense : public Layer<T>
{
public:
    /** Constructs a dense layer for a given input and output size. */
    Dense(int in_size, int out_size)
        : Layer<T>(in_size, out_size)
    {
        prod.resize(in_size, (T)0);
        weights = vec2_type(out_size, vec_type(in_size, (T)0));

        bias.resize(out_size, (T)0);
        sums.resize(out_size, (T)0);
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
    }

    /** Returns the name of this layer. */
    std::string getName() const noexcept override { return "dense"; }

    /** Performs forward propagation for this layer. */
    inline void forward(const T* input, T* out) override
    {
        for(int l = 0; l < Layer<T>::out_size; ++l)
        {
            xsimd::transform(input, &input[Layer<T>::in_size], weights[l].data(), prod.data(),
                [](auto const& a, auto const& b) { return a * b; });

            auto sum = xsimd::reduce(prod.data(), &prod[Layer<T>::in_size], (T)0);
            out[l] = sum + bias[l];
        }
    }

    /**
     * Sets the layer weights from a given vector.
     * 
     * The dimension of the weights vector must be
     * weights[out_size][in_size]
     */
    void setWeights(const std::vector<std::vector<T>>& newWeights)
    {
        for(int i = 0; i < Layer<T>::out_size; ++i)
            for(int k = 0; k < Layer<T>::in_size; ++k)
                weights[i][k] = newWeights[i][k];
    }

    /**
     * Sets the layer weights from a given array.
     * 
     * The dimension of the weights array must be
     * weights[out_size][in_size]
     */
    void setWeights(T** newWeights)
    {
        for(int i = 0; i < Layer<T>::out_size; ++i)
            for(int k = 0; k < Layer<T>::in_size; ++k)
                weights[i][k] = newWeights[i][k];
    }

    /**
     * Sets the layer bias from a given array of size
     * bias[out_size]
     */
    void setBias(T* b)
    {
        for(int i = 0; i < Layer<T>::out_size; ++i)
            bias[i] = b[i];
    }

    /** Returns the weights value at the given indices. */
    T getWeight(int i, int k) const noexcept { return weights[i][k]; }

    /** Returns the bias value at the given index. */
    T getBias(int i) const noexcept { return bias[i]; }

private:
    using vec_type = std::vector<T, XSIMD_DEFAULT_ALLOCATOR(T)>;
    using vec2_type = std::vector<vec_type>;

    vec_type bias;
    vec2_type weights;
    vec_type prod;
    vec_type sums;
};

//====================================================
/**
 * Static implementation of a fully-connected (dense) layer,
 * with no activation.
 */
template <typename T, int in_sizet, int out_sizet>
class DenseT
{
    using v_type = xsimd::simd_type<T>;
    static constexpr auto v_size = (int)v_type::size;
    static constexpr auto v_in_size = ceil_div(in_sizet, v_size);
    static constexpr auto v_out_size = ceil_div(out_sizet, v_size);
    static constexpr auto weights_size = v_in_size * out_sizet;

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = out_sizet;

    DenseT()
    {
        for(int i = 0; i < weights_size; ++i)
            weights[i] = v_type((T)0.0);

        for(int i = 0; i < v_out_size; ++i)
            bias[i] = v_type((T)0.0);

        for(int i = 0; i < v_out_size; ++i)
            outs[i] = v_type((T)0.0);
    }

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "dense"; }

    /** Returns false since dense is not an activation layer. */
    constexpr bool isActivation() const noexcept { return false; }

    /** Reset is a no-op, since Dense does not have state. */
    void reset() { }

    /** Performs forward propagation for this layer. */
    inline void forward(const v_type (&ins)[v_in_size])
    {
        for(int i = 0; i < v_out_size; ++i)
        {
            T out_sum alignas(RTNEURAL_DEFAULT_ALIGNMENT)[v_size] { (T)0 };
            for(int k = 0; k < v_in_size; ++k)
            {
                for(int j = 0; j < v_size; ++j)
                    out_sum[j] += xsimd::hadd(ins[k] * weights[(i * v_size + j) * v_in_size + k]);
            }

            outs[i] = xsimd::load_aligned(out_sum) + bias[i];
        }
    }

    /**
     * Sets the layer weights from a given vector.
     * 
     * The dimension of the weights vector must be
     * weights[out_size][in_size]
     */
    void setWeights(const std::vector<std::vector<T>>& newWeights)
    {
        for(int i = 0; i < out_size; ++i)
        {
            for(int k = 0; k < in_size; ++k)
            {
                auto idx = i * v_in_size + k / v_size;
                weights[idx] = set_value(weights[idx], k % v_size, newWeights[i][k]);
            }
        }
    }

    /**
     * Sets the layer weights from a given vector.
     * 
     * The dimension of the weights array must be
     * weights[out_size][in_size]
     */
    void setWeights(T** newWeights)
    {
        for(int i = 0; i < out_size; ++i)
        {
            for(int k = 0; k < in_size; ++k)
            {
                auto idx = i * v_in_size + k / v_size;
                weights[idx] = set_value(weights[idx], k % v_size, newWeights[i][k]);
            }
        }
    }

    /**
     * Sets the layer bias from a given array of size
     * bias[out_size]
     */
    void setBias(T* b)
    {
        for(int i = 0; i < out_size; ++i)
            bias[i / v_size] = set_value(bias[i / v_size], i % v_size, b[i]);
    }

    v_type outs[v_out_size];

private:
    v_type bias[v_out_size];
    v_type weights[weights_size];
};

template <typename T, int in_sizet>
class DenseT<T, in_sizet, 1>
{
    using v_type = xsimd::simd_type<T>;
    static constexpr auto v_size = (int)v_type::size;
    static constexpr auto v_in_size = ceil_div(in_sizet, v_size);

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = 1;

    DenseT()
    {
        for(int i = 0; i < v_in_size; ++i)
            weights[i] = v_type((T)0.0);

        outs[0] = v_type((T)0.0);
    }

    std::string getName() const noexcept { return "dense"; }
    constexpr bool isActivation() const noexcept { return false; }

    void reset() { }

    inline void forward(const v_type (&ins)[v_in_size])
    {
        T y = (T)0;
        for(int k = 0; k < v_in_size; ++k)
        {
            y += xsimd::hadd(ins[k] * weights[k]);
        }

        outs[0] = v_type(y + bias);
    }

    void setWeights(const std::vector<std::vector<T>>& newWeights)
    {
        for(int i = 0; i < out_size; ++i)
        {
            for(int k = 0; k < in_size; ++k)
            {
                auto idx = k / v_size;
                weights[idx] = set_value(weights[idx], k % v_size, newWeights[i][k]);
            }
        }
    }

    void setWeights(T** newWeights)
    {
        for(int i = 0; i < out_size; ++i)
        {
            for(int k = 0; k < in_size; ++k)
            {
                auto idx = k / v_size;
                weights[idx] = set_value(weights[idx], k % v_size, newWeights[i][k]);
            }
        }
    }

    void setBias(T* b)
    {
        bias = b[0];
    }

    v_type outs[1];

private:
    T bias;
    v_type weights[v_in_size];
};

} // namespace RTNeural

#endif // DENSEXSIMD_H_INCLUDED
