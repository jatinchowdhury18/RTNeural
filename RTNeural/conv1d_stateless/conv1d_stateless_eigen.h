#ifndef CONV1D_STATELESS_EIGEN_H_INCLUDED
#define CONV1D_STATELESS_EIGEN_H_INCLUDED

#include "../Layer.h"
#include "../common.h"
#include <Eigen/Dense>

namespace RTNeural
{
/**
 * Dynamic implementation of a 1-dimensional stateless convolution layer with no activation.
 * This implementation was designed to be used for a single frame of features, fully available at each forward call.
 * So the layer has a NO internal "state"
 *
 * @tparam T Type of the layer (float, double, int ...)
 * @tparam use_bias Whether this layer has a bias
 */
template <typename T, bool use_bias = true>
class Conv1DStateless : public Layer<T>
{
public:
    Conv1DStateless(int in_num_filters_in, int in_num_features_in, int in_num_filters_out, int in_kernel_size, int in_stride);
    Conv1DStateless(std::initializer_list<int> sizes);
    Conv1DStateless(const Conv1DStateless& other);
    Conv1DStateless& operator=(const Conv1DStateless& other);
    virtual ~Conv1DStateless() = default;

    /** Resets the layer state. */
    void reset() override {};

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "conv1d_stateless"; }

    /** Returns false since convolution is not an activation layer. */
    constexpr bool isActivation() const noexcept { return false; }

    /** Performs forward propagation for this layer. */
    inline void forward(const T* input, T* output) noexcept override
    {
        auto inMatrix = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
            RTNeuralEigenAlignment>(input, num_filters_in, num_features_in);

        auto outMatrix = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
            RTNeuralEigenAlignment>(output, num_filters_out, num_features_out);

        // perform a multichannel convolution
        for(int i = 0; i < num_filters_out; i++)
        {
            for(int j = 0; j < num_features_out; j++)
            {
                if (use_bias)
                    outMatrix(i, j) = kernelWeights[i].cwiseProduct(inMatrix.middleCols(j * stride, kernel_size)).sum() + bias(i);
                else
                    outMatrix(i, j) = kernelWeights[i].cwiseProduct(inMatrix.middleCols(j * stride, kernel_size)).sum();
            }
        }
    }

    /**
     * Sets the layer weights.
     *
     * The weights vector must have size weights[num_filters_out][num_filters_in][kernel_size]
     */
    void setWeights(const std::vector<std::vector<std::vector<T>>>& inWeights);

    /**
     * Sets the layer biases.
     *
     * The bias vector must have size bias[num_filters_out]
     */
    void setBias(const std::vector<T>& inBias);

    /** Returns the size of the convolution kernel. */
    int getKernelSize() const noexcept { return kernel_size; }

    /** Returns the stride. */
    int getStride() const noexcept { return stride; }

private:
    const int num_filters_in;
    const int num_features_in;
    const int num_filters_out;
    const int kernel_size;
    const int stride;
    const int num_features_out = (num_features_in - kernel_size) / stride + 1;

    std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> kernelWeights;
    Eigen::Vector<T, Eigen::Dynamic> bias;
};

//====================================================

/**
 * Static implementation of a 1-dimensional stateless convolution layer with no activation.
 * This implementation was designed to be used for a single frame of features, fully available at each forward call.
 * So the layer has a NO internal "state"
 *
 * @tparam T Type of the layer (float, double, int ...)
 * @tparam num_filters_in number of input filters
 * @tparam num_features_in number of input features
 * @tparam num_filters_out number of output filters
 * @tparam kernel_size size of the convolution kernel
 * @tparam stride convolution stride
 */
template <typename T, int num_filters_in, int num_features_in, int num_filters_out, int kernel_size,
    int stride, bool use_bias = true>
class Conv1DStatelessT
{
    using bias_type = Eigen::Vector<T, use_bias ? num_filters_out : 0>;
    using weights_type = Eigen::Matrix<T, num_filters_in, kernel_size>;
    using input_type = Eigen::Matrix<T, num_filters_in, num_features_in>;
    static constexpr int num_features_out = (num_features_in - kernel_size) / stride + 1; // TODO: to test
    using output_type = Eigen::Matrix<T, num_filters_out, num_features_out>;

public:
    Conv1DStatelessT();

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "conv1d_stateless"; }

    /** Returns false since convolution is not an activation layer. */
    constexpr bool isActivation() const noexcept { return false; }

    /** Empty function, this layer has no state */
    void reset() {};

    /** Performs forward propagation for this layer. */
    inline void forward(const input_type& inMatrix) noexcept
    {
        // perform a multichannel convolution
        for(int i = 0; i < num_filters_out; i++)
        {
            for(int j = 0; j < num_features_out; j++)
            {
                // TODO: manage to use middleCols<kernel_size>(j*stride)
                if (use_bias)
                    outs(i, j) = weights[i].cwiseProduct(inMatrix.middleCols(j * stride, kernel_size)).sum() + bias(i);
                else
                    outs(i, j) = weights[i].cwiseProduct(inMatrix.middleCols(j * stride, kernel_size)).sum();
            }
        }
    }

    /**
     * Sets the layer weights.
     *
     * The weights vector must have size weights[num_filters_out][num_filters_in][kernel_size]
     */
    void setWeights(const std::vector<std::vector<std::vector<T>>>& inWeights);

    /**
     * Sets the layer biases.
     *
     * The bias vector must have size bias[num_filters_out]
     */
    void setBias(const std::vector<T>& inBias);

    /** Returns the size of the convolution kernel. */
    int getKernelSize() const noexcept { return kernel_size; }

    /** Returns the convolution dilation rate. */
    int getStride() const noexcept { return stride; }

    Eigen::Map<output_type, RTNeuralEigenAlignment> outs;

private:
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[num_filters_out * num_features_out];

    weights_type weights[num_filters_out];
    bias_type bias;
};

} // RTNEURAL

#endif // CONV1D_STATELESS_EIGEN_H_INCLUDED
