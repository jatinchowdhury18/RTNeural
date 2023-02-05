#include "conv1d_stateless_eigen.h"

namespace RTNeural
{
template <typename T, bool use_bias>
Conv1DStateless<T, use_bias>::Conv1DStateless(int in_num_filters_in, int in_num_features_in, int in_num_filters_out, int in_kernel_size, int in_stride)
    : num_filters_in(in_num_filters_in)
    , num_features_in(in_num_features_in)
    , num_filters_out(in_num_filters_out)
    , kernel_size(in_kernel_size)
    , stride(in_stride)
{
    kernelWeights.resize(num_filters_out);
    for(int i = 0; i < num_filters_out; ++i)
        kernelWeights[i] = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(num_features_in, kernel_size);

    if (use_bias)
        bias = Eigen::Vector<T, Eigen::Dynamic>::Zero(num_filters_out);
}

template <typename T, bool use_bias>
Conv1DStateless<T, use_bias>::Conv1DStateless(std::initializer_list<int> sizes)
    : Conv1DStateless<T, use_bias>(*sizes.begin(), *(sizes.begin() + 1), *(sizes.begin() + 2), *(sizes.begin() + 3), *(sizes.begin() + 4))
{
}

template <typename T, bool use_bias>
Conv1DStateless<T, use_bias>::Conv1DStateless(const Conv1DStateless& other)
    : Conv1DStateless(other.num_filters_in, other.num_features_in, other.num_filters_out, other.kernel_size, other.stride)
{
}

template <typename T, bool use_bias>
Conv1DStateless<T, use_bias>& Conv1DStateless<T, use_bias>::operator=(const Conv1DStateless<T, use_bias>& other)
{
    return *this = Conv1DStateless<T>(other);
}

template <typename T, bool use_bias>
void Conv1DStateless<T, use_bias>::setWeights(const std::vector<std::vector<std::vector<T>>>& inWeights)
{
    for(int i = 0; i < Layer<T>::out_size; ++i)
        for(int k = 0; k < Layer<T>::in_size; ++k)
            for(int j = 0; j < kernel_size; ++j)
                kernelWeights[i](k, j) = inWeights[i][k][j];
}

template <typename T, bool use_bias>
void Conv1DStateless<T, use_bias>::setBias(const std::vector<T>& inBias)
{
    if(use_bias)
    {
        for(int i = 0; i < Layer<T>::out_size; ++i)
            bias(i) = inBias[i];
    }
}

//====================================================

template <typename T, int num_filters_in, int num_features_in, int num_filters_out, int kernel_size, int stride, bool use_bias>
Conv1DStatelessT<T, num_filters_in, num_features_in, num_filters_out, kernel_size, stride, use_bias>::Conv1DStatelessT()
    : outs(outs_internal)
{
    for(int k = 0; k < num_filters_out; ++k)
        weights[k] = weights_type::Zero();

    if(use_bias)
        bias = bias_type::Zero();
}

template <typename T, int num_filters_in, int num_features_in, int num_filters_out, int kernel_size, int stride, bool use_bias>
void Conv1DStatelessT<T, num_filters_in, num_features_in, num_filters_out, kernel_size, stride, use_bias>::setWeights(const std::vector<std::vector<std::vector<T>>>& inWeights)
{
    for(int i = 0; i < num_features_out; ++i)
        for(int k = 0; k < num_features_in; ++k)
            for(int j = 0; j < kernel_size; ++j)
                weights[i](k, j) = inWeights[i][k][j];
}

template <typename T, int num_filters_in, int num_features_in, int num_filters_out, int kernel_size, int stride, bool use_bias>
void Conv1DStatelessT<T, num_filters_in, num_features_in, num_filters_out, kernel_size, stride, use_bias>::setBias(const std::vector<T>& inBias)
{
    if(use_bias)
        for(int i = 0; i < num_features_out; ++i)
            bias(i) = inBias[i];
}
} // RTNeural
