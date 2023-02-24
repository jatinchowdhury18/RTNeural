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
 */
template <typename T>
class Conv1DStateless : public Layer<T>
{
public:
    Conv1DStateless(int in_num_filters_in, int in_num_features_in, int in_num_filters_out, int in_kernel_size, int in_stride, bool in_valid_pad);
    Conv1DStateless(std::initializer_list<int> sizes);
    Conv1DStateless(const Conv1DStateless& other);
    Conv1DStateless& operator=(const Conv1DStateless& other);
    virtual ~Conv1DStateless() = default;

    static int computeNumFeaturesOut(int num_features_in, int kernel_size, int stride, int valid_pad)
    {
        // Based on tensorflow docs: https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
        if(valid_pad)
            return std::ceil(static_cast<float>(num_features_in - kernel_size + 1) / static_cast<float>(stride));

        return std::ceil(static_cast<float>(num_features_in) / static_cast<float>(stride));
    }

    /** Resets the layer state. */
    void reset() override {};

    /** Returns the name of this layer. */
    std::string getName() const noexcept override { return "conv1d_stateless"; }

    /** Returns false since convolution is not an activation layer. */
    constexpr bool isActivation() const noexcept { return false; }

    /** Performs forward propagation for this layer. */
    inline void forward(const T* input, T* output) noexcept override
    {
        auto inMatrix = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
            RTNeuralEigenAlignment>(input, num_filters_in, num_features_in);

        auto outMatrix = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
            RTNeuralEigenAlignment>(output, num_filters_out, num_features_out);

        if(valid_pad)
        {
            for(int i = 0; i < num_filters_out; i++)
                for(int j = 0; j < num_features_out; j++)
                    outMatrix(i, j) += kernelWeights[i].cwiseProduct(inMatrix.middleCols(j * stride, kernel_size)).sum();
        }
        else
        {
            for(int i = 0; i < num_filters_out; i++)
            {
                int j = 0;

                for(; j * stride < pad_left; j++)
                {
                    const int eff_kernel_size = kernel_size - pad_left + j * stride;
                    outMatrix(i, j) += kernelWeights[i].rightCols(eff_kernel_size).cwiseProduct(inMatrix.leftCols(eff_kernel_size)).sum();
                }

                for(; j * stride - pad_left + kernel_size < num_features_in; j++)
                    outMatrix(i, j) += kernelWeights[i].cwiseProduct(inMatrix.middleCols(j * stride - pad_left, kernel_size)).sum();

                for(; j * stride - pad_left + kernel_size <= num_features_in + pad_right; j++)
                {
                    const int eff_kernel_size = num_features_in - (j * stride - pad_left);
                    outMatrix(i, j) += kernelWeights[i].leftCols(eff_kernel_size).cwiseProduct(inMatrix.rightCols(eff_kernel_size)).sum();
                }
            }
        }
    }

    /**
     * Sets the layer weights.
     *
     * The weights vector must have size weights[num_filters_out][num_filters_in][kernel_size]
     */
    void setWeights(const std::vector<std::vector<std::vector<T>>>& inWeights);

    /** Returns the size of the convolution kernel. */
    int getKernelSize() const noexcept { return kernel_size; }

    /** Returns the stride. */
    int getStride() const noexcept { return stride; }

    void printWeights() const noexcept
    {
        for(int i = 0; i < num_filters_out; i++)
            std::cout << kernelWeights[i] << std::endl;
    }

private:
    const int num_filters_in;
    const int num_features_in;
    const int num_filters_out;
    const int kernel_size;
    const int stride;
    const int num_features_out;
    const bool valid_pad;
    int pad_left = 0;
    int pad_right = 0;

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
    int stride>
class Conv1DStatelessT
{
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

    /** Returns the size of the convolution kernel. */
    int getKernelSize() const noexcept { return kernel_size; }

    /** Returns the convolution dilation rate. */
    int getStride() const noexcept { return stride; }

    Eigen::Map<output_type, RTNeuralEigenAlignment> outs;

private:
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[num_filters_out * num_features_out];

    int pad_left;
    int pad_right;

    weights_type weights[num_filters_out];
};

} // RTNEURAL

#endif // CONV1D_STATELESS_EIGEN_H_INCLUDED
