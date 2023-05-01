#ifndef CONV1DXSIMD_H_INCLUDED
#define CONV1DXSIMD_H_INCLUDED

#include "../Layer.h"
#include "../common.h"
#include <numeric>
#include <vector>

namespace RTNeural
{

/**
 * Dynamic implementation of a 1-dimensional convolution layer
 * with no activation.
 *
 * This implementation was designed to be used for "temporal
 * convolution", so the layer has a "state" made up of past inputs
 * to the layer. To ensure that the state is initialized to zero,
 * please make sure to call `reset()` before your first call to
 * the `forward()` method.
 */
template <typename T>
class Conv1D : public Layer<T>
{
public:
    /**
     * Constructs a convolution layer for the given dimensions.
     *
     * @param in_size: the input size for the layer
     * @param out_size: the output size for the layer
     * @param kernel_size: the size of the convolution kernel
     * @param dilation: the dilation rate to use for dilated convolution
     */
    Conv1D(int in_size, int out_size, int kernel_size, int dilation);
    Conv1D(std::initializer_list<int> sizes);
    Conv1D(const Conv1D& other);
    Conv1D& operator=(const Conv1D& other);
    virtual ~Conv1D();

    /** Resets the layer state. */
    void reset() override;

    /** Returns the name of this layer. */
    std::string getName() const noexcept override { return "conv1d"; }

    /** Performs forward propagation for this layer. */
    inline void forward(const T* input, T* h) noexcept override
    {
        // insert input into a circular buffer
        vCopy(input, state[state_ptr].data(), Layer<T>::in_size);

        // set state pointers to particular columns of the buffer
        setStatePointers();

        // copy selected columns to a helper variable
        for(int k = 0; k < kernel_size; ++k)
        {
            const auto& col = state[state_ptrs[k]];
            vCopy(col.data(), state_cols[k].data(), Layer<T>::in_size);
        }

        // perform multi-channel convolution
        vCopy(bias.data(), h, Layer<T>::out_size);
        for(int i = 0; i < Layer<T>::out_size; ++i)
        {
            for(int k = 0; k < kernel_size; ++k)
                h[i] += vMult(weights[i][k].data(), state_cols[k].data(), prod_state.data(), Layer<T>::in_size);
        }

        state_ptr = (state_ptr == state_size - 1 ? 0 : state_ptr + 1); // iterate state pointer forwards
    }

    /**
     * Sets the layer weights.
     *
     * The weights vector must have size weights[out_size][in_size][kernel_size * dilation]
     */
    void setWeights(const std::vector<std::vector<std::vector<T>>>& weights);

    /**
     * Sets the layer biases.
     *
     * The bias vector must have size bias[out_size]
     */
    void setBias(const std::vector<T>& biasVals);

    /** Returns the size of the convolution kernel. */
    int getKernelSize() const noexcept { return kernel_size; }

    /** Returns the convolution dilation rate. */
    int getDilationRate() const noexcept { return dilation_rate; }

private:
    using vec_type = std::vector<T, xsimd::aligned_allocator<T>>;
    using vec2_type = std::vector<vec_type>;
    using vec3_type = std::vector<vec2_type>;

    const int dilation_rate;
    const int kernel_size;
    const int state_size;

    vec3_type weights;
    vec_type bias;

    vec2_type state;
    vec2_type state_cols;

    int state_ptr = 0;
    std::vector<int> state_ptrs;

    vec_type prod_state;

    /** Sets pointers to state array columns. */
    inline void setStatePointers()
    {
        for(int k = 0; k < kernel_size; ++k)
            state_ptrs[k] = (state_ptr + state_size - k * dilation_rate) % state_size;
    }
};

//====================================================
/**
 * Static implementation of a 1-dimensional convolution layer
 * with no activation.
 *
 * This implementation was designed to be used for "temporal
 * convolution", so the layer has a "state" made up of past inputs
 * to the layer. To ensure that the state is initialized to zero,
 * please make sure to call `reset()` before your first call to
 * the `forward()` method.
 *
 * @param in_sizet: the input size for the layer
 * @param out_sizet: the output size for the layer
 * @param kernel_size: the size of the convolution kernel
 * @param dilation_rate: the dilation rate to use for dilated convolution
 * @param dynamic_state: use dynamically allocated layer state
 */
template <typename T, int in_sizet, int out_sizet, int kernel_size, int dilation_rate, bool dynamic_state = false>
class Conv1DT
{
    using v_type = xsimd::simd_type<T>;
    static constexpr auto v_size = (int)v_type::size;
    static constexpr auto state_size = (kernel_size - 1) * dilation_rate + 1;
    static constexpr auto v_in_size = ceil_div(in_sizet, v_size);
    static constexpr auto v_out_size = ceil_div(out_sizet, v_size);

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = out_sizet;

    Conv1DT();

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "conv1d"; }

    /** Returns false since convolution is not an activation layer. */
    constexpr bool isActivation() const noexcept { return false; }

    /** Resets the layer state. */
    void reset();

    /** Performs forward propagation for this layer. */
    template <int DR = dilation_rate>
    inline typename std::enable_if<(DR > 1), void>::type
    forward(const v_type (&ins)[v_in_size]) noexcept
    {
        // insert input into a circular buffer
        std::copy(std::begin(ins), std::end(ins), state[state_ptr].begin());

        // set state pointers to particular columns of the buffer
        setStatePointers();

        // copy selected columns to a helper variable
        for(int k = 0; k < kernel_size; ++k)
        {
            const auto& col = state[state_ptrs[k]];
            std::copy(col.begin(), col.end(), state_cols[k].begin());
        }

        // perform multi-channel convolution
        for(int i = 0; i < v_out_size; ++i)
        {
            alignas(RTNEURAL_DEFAULT_ALIGNMENT) T out_sum[v_size] {};
            for(int k = 0; k < v_size && (i * v_size + k) < out_size; ++k)
            {
                assert(i * v_size + k < out_size);
                const auto& subWeights = weights[i * v_size + k];
                v_type accum {};
                for(int j = 0; j < kernel_size; ++j)
                {
                    accum += std::inner_product(
                        subWeights[j].begin(),
                        subWeights[j].end(),
                        state_cols[j].begin(),
                        v_type {});
                }
                out_sum[k] = xsimd::reduce_add(accum);
            }

            outs[i] = xsimd::load_aligned(out_sum) + bias[i];
        }

        state_ptr = (state_ptr == state_size - 1 ? 0 : state_ptr + 1); // iterate state pointer forwards
    }

    /** Performs forward propagation for this layer. */
    template <int DR = dilation_rate, int KS = kernel_size>
    inline typename std::enable_if<(DR == 1 && KS > 1), void>::type
    forward(const v_type (&ins)[v_in_size]) noexcept
    {
        // insert input into a circular buffer
        std::copy(std::begin(ins), std::end(ins), state[state_ptr].begin());

        // set state pointers to particular columns of the buffer
        setStatePointers();

        // perform multi-channel convolution
        for(int i = 0; i < v_out_size; ++i)
        {
            alignas(RTNEURAL_DEFAULT_ALIGNMENT) T out_sum[v_size] {};
            for(int k = 0; k < v_size && (i * v_size + k) < out_size; ++k)
            {
                const auto& subWeights = weights[i * v_size + k];
                v_type accum {};
                for(int j = 0; j < kernel_size; ++j)
                {
                    accum += std::inner_product(
                        subWeights[j].begin(),
                        subWeights[j].end(),
                        state[(state_ptr + state_size - j) % state_size].begin(),
                        v_type {});
                }
                out_sum[k] = xsimd::reduce_add(accum);
            }

            outs[i] = xsimd::load_aligned(out_sum) + bias[i];
        }

        state_ptr = (state_ptr == state_size - 1 ? 0 : state_ptr + 1); // iterate state pointer forwards
    }

    /** Performs forward propagation for this layer. */
    template <int DR = dilation_rate, int KS = kernel_size>
    inline typename std::enable_if<DR == 1 && KS == 1, void>::type
    forward(const v_type (&ins)[v_in_size]) noexcept
    {
        for(int i = 0; i < v_out_size; ++i)
        {
            alignas(RTNEURAL_DEFAULT_ALIGNMENT) T out_sum[v_size] {};
            for(int k = 0; k < v_size && (i * v_size + k) < out_size; ++k)
            {
                const auto& subWeights = weights[i * v_size + k][0];

                v_type accum {};
                for(int j = 0; j < v_in_size; ++j)
                    accum += subWeights[j] * ins[j];
                out_sum[k] = xsimd::reduce_add(accum);
            }

            outs[i] = xsimd::load_aligned(out_sum) + bias[i];
        }
    }

    /**
     * Sets the layer weights.
     *
     * The weights vector must have size weights[out_size][in_size][kernel_size * dilation]
     */
    void setWeights(const std::vector<std::vector<std::vector<T>>>& weights);

    /**
     * Sets the layer biases.
     *
     * The bias vector must have size bias[out_size]
     */
    void setBias(const std::vector<T>& biasVals);

    /** Returns the size of the convolution kernel. */
    int getKernelSize() const noexcept { return kernel_size; }

    /** Returns the convolution dilation rate. */
    int getDilationRate() const noexcept { return dilation_rate; }

    v_type outs[v_out_size];

private:
    template <int DS = dynamic_state>
    typename std::enable_if<DS, void>::type resize_state()
    {
        state.resize(state_size, {});
    }

    template <int DS = dynamic_state>
    typename std::enable_if<!DS, void>::type resize_state() { }

    using state_col_type = std::array<v_type, v_in_size>;
    using state_type = typename std::conditional<dynamic_state, std::vector<state_col_type, xsimd::aligned_allocator<state_col_type>>, std::array<state_col_type, state_size>>::type;
    using weights_type = std::array<std::array<v_type, v_in_size>, kernel_size>;

    state_type state {};
    weights_type state_cols {};

    int state_ptr = 0;
    std::array<int, kernel_size> state_ptrs {};

    weights_type weights[out_size] {};
    v_type bias[v_out_size] {};

    /** Sets pointers to state array columns. */
    inline void setStatePointers()
    {
        for(int k = 0; k < kernel_size; ++k)
            state_ptrs[k] = (state_ptr + state_size - k * dilation_rate) % state_size;
    }
};
} // namespace RTNeural

#endif // CONV1DXSIMD_H_INCLUDED
