#ifndef CONV2D_EIGEN_H_INCLUDED
#define CONV2D_EIGEN_H_INCLUDED

#include "Layer.h"

#include "../Layer.h"
#include "../common.h"
#include "conv1d_stateless/conv1d_stateless.h"
#include <Eigen/Dense>

namespace RTNeural
{
/**
 * Dynamic implementation of a 2-dimensional convolution layer with no activation.
 *
 * @tparam T Type of the layer (float, double, int ...)
 * @tparam num_filters_in number of input filters
 * @tparam num_filters_out number of output filters
 * @tparam num_features_in number of input features
 * @tparam kernel_size_time size of the convolution kernel (time axis)
 * @tparam kernel_size_feature size of the convolution kernel (feature axis)
 * @tparam dilation_rate dilation_rate (time axis)
 * @tparam stride convolution stride (feature axis)
 */
template <typename T>
class Conv2D : public Layer<T>
{
public:
    /**
     * @param in_num_filters_in number of input filters
     * @param in_num_filters_out number of output filters
     * @param in_num_features_in number of input features
     * @param in_kernel_size_time size of the convolution kernel (time axis)
     * @param in_kernel_size_feature size of the convolution kernel (feature axis)
     * @param in_dilation_rate dilation_rate (time axis)
     * @param in_stride convolution stride (feature axis)
     */
    Conv2D(int in_num_filters_in, int in_num_filters_out, int in_num_features_in, int in_kernel_size_time, int in_kernel_size_feature, int in_dilation_rate, int in_stride);
    Conv2D(std::initializer_list<int> sizes);
    Conv2D(const Conv2D& other);
    Conv2D& operator=(const Conv2D& other);
    virtual ~Conv2D() = default;

    /** Reset the layer's state */
    void reset() override
    {
        state_index = 0;

        for(int i = 0; i < receptive_field; i++)
        {
            state[i].setZero();
        }
    };

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "conv2d"; }

    /** Returns false since convolution is not an activation layer. */
    constexpr bool isActivation() const noexcept { return false; }

    /** Performs forward propagation for this layer. */
    inline void forward(const T* input, T* output) noexcept override
    {
        auto inMatrix = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
            RTNeuralEigenAlignment>(input, num_filters_in, num_features_in);

        auto outMatrix = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
            RTNeuralEigenAlignment>(output, num_filters_out, num_features_out);

        for(int i = 0; i < kernel_size_time; i++)
        {
            conv1dLayers[i].forward(inMatrix);
            state[(state_index + dilation_rate * i) % receptive_field] += conv1dLayers[i].outs;
        }

        outMatrix = state[state_index + dilation_rate * (kernel_size_time - 1)] + bias;
        state[state_index].setZero();
        state_index = state_index == receptive_field - 1 ? 0 : state_index + 1;
    }

    /**
     * Sets the layer weights.
     *
     * The weights vector must have size weights[num_filters_out][num_filters_in][kernel_size]
     */
    void setWeights(const std::vector<std::vector<std::vector<std::vector<T>>>>& inWeights);

    /**
     * Sets the layer biases.
     *
     * The bias vector must have size bias[num_filters_out]
     */
    void setBias(const std::vector<T>& inBias);

    /** Returns the size of the convolution kernel (time axis). */
    int getKernelSizeTime() const noexcept { return kernel_size_time; }

    /** Returns the size of the convolution kernel (feature axis). */
    int getKernelSizeFeature() const noexcept { return kernel_size_feature; }

    /** Returns the convolution stride (feature axis) */
    int getStride() const noexcept { return stride; }

    /** Returns the convolution dilation rate (time axis) */
    int getDilationRate() const noexcept { return dilation_rate; }

private:
    const int num_filters_in;
    const int num_features_in;
    const int num_filters_out;
    const int kernel_size_time;
    const int kernel_size_feature;
    const int dilation_rate;
    const int stride;
    const int num_features_out;
    const int receptive_field;

    std::vector<Conv1DStateless<T, false>> conv1dLayers;

    std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> state;

    int state_index = 0;

    Eigen::Vector<T, Eigen::Dynamic> bias;
};

//====================================================

/**
 * Static implementation of a 2-dimensional convolution layer with no activation.
 *
 * @tparam T Type of the layer (float, double, int ...)
 * @tparam num_filters_in number of input filters
 * @tparam num_filters_out number of output filters
 * @tparam num_features_in number of input features
 * @tparam kernel_size_time size of the convolution kernel (time axis)
 * @tparam kernel_size_feature size of the convolution kernel (feature axis)
 * @tparam dilation_rate dilation_rate (time axis)
 * @tparam stride convolution stride (feature axis)
 */
template <typename T, int num_filters_in, int num_filters_out, int num_features_in, int kernel_size_time,
    int kernel_size_feature, int dilation_rate, int stride>
class Conv2DT
{
    using bias_type = Eigen::Vector<T, num_filters_out>;
    using input_type = Eigen::Matrix<T, num_filters_in, num_features_in>;
    static constexpr int num_features_out = (num_features_in - kernel_size_feature) / stride + 1; // TODO: to test
    using output_type = Eigen::Matrix<T, num_filters_out, num_features_out>;
    static constexpr int receptive_field = 1 + (kernel_size_time - 1) * dilation_rate;

public:
    Conv2DT() = default;

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "conv2d"; }

    /** Returns false since convolution is not an activation layer. */
    constexpr bool isActivation() const noexcept { return false; }

    /** Reset the layer's state */
    void reset()
    {
        state_index = 0;

        for(int i = 0; i < receptive_field; i++)
        {
            state[i] = output_type::Zero();
        }
    };

    /** Performs forward propagation for this layer. */
    inline void forward(const input_type& inMatrix) noexcept
    {
        for(int i = 0; i < kernel_size_time; i++)
        {
            conv1dLayers[i].forward(inMatrix);
            state[(state_index + dilation_rate * i) % receptive_field] += conv1dLayers[i].outs;
        }

        outs = state[state_index + dilation_rate * (kernel_size_time - 1)] + bias;
        state[state_index] = output_type::Zero();
        state_index = state_index == receptive_field - 1 ? 0 : state_index + 1;
    }

    /**
     * Sets the layer weights.
     *
     * The weights vector must have size weights [kernel_size_time][num_filters_out][num_filters_in][kernel_size_feature]
     */
    void setWeights(const std::vector<std::vector<std::vector<std::vector<T>>>>& inWeights);

    /**
     * Sets the layer biases.
     *
     * The bias vector must have size bias[num_filters_out]
     */
    void setBias(const std::vector<T>& inBias);

    /** Returns the size of the convolution kernel (time axis). */
    int getKernelSizeTime() const noexcept { return kernel_size_time; }

    /** Returns the size of the convolution kernel (feature axis). */
    int getKernelSizeFeature() const noexcept { return kernel_size_feature; }

    /** Returns the convolution stride */
    int getStride() const noexcept { return stride; }

    /** Returns the convolution dilation rate */
    int getDilationRate() const noexcept { return dilation_rate; }

    Eigen::Map<output_type, RTNeuralEigenAlignment> outs;

private:
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[num_filters_out * num_features_out];

    std::array<Conv1DStatelessT<T, num_filters_in, num_features_in, num_filters_out, kernel_size_feature, stride, false>,
        kernel_size_time>
        conv1dLayers;

    std::array<output_type, receptive_field> state;

    int state_index = 0;

    bias_type bias;
};

} // RTNEURAL

#endif // CONV2D_EIGEN_H_INCLUDED
