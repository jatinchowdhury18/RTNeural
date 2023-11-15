#ifndef CONV1D_H_INCLUDED
#define CONV1D_H_INCLUDED

#if RTNEURAL_USE_EIGEN
#include "conv1d_eigen.h"
#include "conv1d_eigen.tpp"
#elif RTNEURAL_USE_XSIMD
#include "conv1d_xsimd.h"
#include "conv1d_xsimd.tpp"
#else
#include "../Layer.h"
#include "../common.h"
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
class Conv1D final : public Layer<T>
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
        std::copy(input, input + Layer<T>::in_size, state[state_ptr]);

        // set state pointers to particular columns of the buffer
        setStatePointers();

        // copy selected columns to a helper variable
        for(int k = 0; k < kernel_size; ++k)
        {
            const auto& col = state[state_ptrs[k]];
            std::copy(col, col + Layer<T>::in_size, state_cols[k]);
        }

        // perform multi-channel convolution
        for(int i = 0; i < Layer<T>::out_size; ++i)
        {
            h[i] = bias[i];
            for(int k = 0; k < kernel_size; ++k)
                h[i] = std::inner_product(
                    weights[i][k],
                    weights[i][k] + Layer<T>::in_size,
                    state_cols[k],
                    h[i]);
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
    const int dilation_rate;
    const int kernel_size;
    const int state_size;

    T*** weights;
    T* bias;

    T** state;
    T** state_cols;

    int* state_ptrs;
    int state_ptr = 0;

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
 * @param groups_of: controls connections between inputs and outputs
 */
template <typename T, int in_sizet, int out_sizet, int kernel_size, int dilation_rate, int groups_of = 1, bool dynamic_state = false>
class Conv1DT
{
    static_assert((in_sizet % groups_of == 0) && (out_sizet % groups_of == 0), "in_sizet and out_sizet must be divisible by groups_of!");

    static constexpr auto state_size = (kernel_size - 1) * dilation_rate + 1;

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = out_sizet;
    static constexpr auto group_count = in_size / groups_of;

    Conv1DT();

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "conv1d"; }

    /** Returns false since convolution is not an activation layer. */
    constexpr bool isActivation() const noexcept { return false; }

    /** Resets the layer state. */
    void reset();

    /** Performs forward propagation for this layer. */
    inline void forward(const T (&ins)[in_size]) noexcept
    {
        // insert input into a circular buffer
        std::copy(std::begin(ins), std::end(ins), state[state_ptr].begin());

        // set state pointers to particular columns of the buffer
        setStatePointers();

        // perform multi-channel convolution
        for(int i = 0; i < out_size; ++i)
        {
            outs[i] = bias[i];

            // copy selected columns to a helper variable
            for(int k = 0; k < kernel_size; ++k)
            {
                const auto& column = state[state_ptrs[k]];
                const auto column_begin = column.begin() + i * group_count;
                const auto column_end = column.begin() + i * group_count + group_count;
                std::copy(column_begin, column_end, state_cols[k].begin());
            }

            for(int k = 0; k < kernel_size; ++k)
            {
                outs[i] = std::inner_product(
                    weights[i][k].begin(),
                    weights[i][k].end(),
                    state_cols[k].begin(),
                    outs[i]);
            }
        }

        state_ptr = (state_ptr == state_size - 1 ? 0 : state_ptr + 1); // iterate state pointer forwards
    }

    /**
     * Sets the layer weights.
     *
     * The weights vector must have size weights[out_size][group_count][kernel_size * dilation]
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

    T outs alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];

private:
    template <int DS = dynamic_state>
    typename std::enable_if<DS, void>::type resize_state()
    {
        state.resize(state_size, {});
    }

    template <int DS = dynamic_state>
    typename std::enable_if<!DS, void>::type resize_state() { }

    // The `state_type` is a matrix with the shape (in_size, state_size), where
    // `state_size` corresponds to the width of the kernel given its size and
    // dilation rate. Originally, during inference, samples from all input
    // channels are copied onto the state. You can think of the state as a
    // buffer of samples where each channel is stacked upon each other.
    //
    // The `weights_type` is a matrix with the shape (in_size, kernel_size),
    // which corresponds to the groups of kernels used to produce the output.
    // It's used in two fields, `weights`, which extends its dimension by
    // associating a `weights_type` with each channel by `weights[out_size]`;
    // `state_cols` on the other hand is an auxiliary field which is used
    // as an intermediate structure for computing the output. Specifically,
    // for a given cell in the kernel, it copies the corresponding column
    // in the state. Note that the `state_ptrs` function is used to store
    // which columns to actually get from, since we're taking into account
    // the dilation.
    //
    // Inference is performed by iterating over the output channels. For
    // each output channel, maps a group of kernels. The `outs` array is
    // first set with the `bias`, then, it's set to the dot product of
    // the `weights` and the `state_cols` at a given kernel cell.
    //
    // In the case of a `groups_of` parameter other than `1`, the `weights_type`
    // would now have the shape `(in_size / groups_of, kernel_size)`, the idea
    // being that the input channels are now treated as if they are being processed
    // by different convolutions, for instance, a 6in->3out convolution in groups
    // of 3 channels would be processed as if it had 2 3in->3out convolutions.
    //
    // Meanwhile, the `state_type` can remain the same shape as we still retain
    // the same number of input channels, just that only the input channels for
    // the current group being processed is copied over!

    using state_type = std::array<std::array<T, in_size>, state_size>;
    using weights_type = std::array<std::array<T, group_count>, kernel_size>;

    alignas(RTNEURAL_DEFAULT_ALIGNMENT) state_type state;
    alignas(RTNEURAL_DEFAULT_ALIGNMENT) weights_type state_cols;

    int state_ptr = 0;
    std::array<int, kernel_size> state_ptrs;

    alignas(RTNEURAL_DEFAULT_ALIGNMENT) weights_type weights[out_size];
    alignas(RTNEURAL_DEFAULT_ALIGNMENT) std::array<T, out_size> bias;

    /** Sets pointers to state array columns. */
    inline void setStatePointers()
    {
        for(int k = 0; k < kernel_size; ++k)
            state_ptrs[k] = (state_ptr + state_size - k * dilation_rate) % state_size;
    }
};
} // namespace RTNeural
#endif
#endif // CONV1D_H_INCLUDED
