#ifndef CONV1DEIGEN_H_INCLUDED
#define CONV1DEIGEN_H_INCLUDED

#include "../Layer.h"
#include <Eigen/Dense>

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
    virtual inline void forward(const T* input, T* h) override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::Aligned16>(
            input, Layer<T>::in_size, 1);

        // insert input into double-buffered state
        state.col(state_ptr) = inVec;
        state.col(state_ptr + state_size) = inVec;

        for(int i = 0; i < Layer<T>::out_size; ++i)
            outVec(i, 0) = state.block(0, state_ptr, Layer<T>::in_size, state_size).cwiseProduct(kernelWeights[i]).sum();

        outVec = outVec + bias;
        std::copy(outVec.data(), outVec.data() + Layer<T>::out_size, h);

        state_ptr = (state_ptr == 0 ? state_size - 1 : state_ptr - 1); // iterate state pointer in reverse
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

    std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> kernelWeights;
    Eigen::Matrix<T, Eigen::Dynamic, 1> bias;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> state;
    int state_ptr = 0;

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
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
 */
template <typename T, int in_sizet, int out_sizet, int kernel_size, int dilation_rate>
class Conv1DT
{
    using vec_type = Eigen::Matrix<T, out_sizet, 1>;

    static constexpr auto state_size = kernel_size * dilation_rate;
    using state_type = Eigen::Matrix<T, in_sizet, 2 * state_size>;

    using weights_type = Eigen::Matrix<T, in_sizet, state_size>;

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
    inline void forward(const Eigen::Matrix<T, in_size, 1>& ins)
    {
        // insert input into double-buffered state
        state.col(state_ptr) = ins;
        state.col(state_ptr + state_size) = ins;

        for(int i = 0; i < out_size; ++i)
            outs(i) = state.block(0, state_ptr, in_size, state_size).cwiseProduct(weights[i]).sum();

        outs = outs + bias;

        state_ptr = (state_ptr == 0 ? state_size - 1 : state_ptr - 1); // iterate state pointer in reverse
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

    Eigen::Map<vec_type, Eigen::Aligned16> outs;

private:
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];

    state_type state;
    int state_ptr = 0;

    weights_type weights[out_size];
    vec_type bias;
};

} // RTNeural

#endif // CONV1DEIGEN_H_INCLUDED
