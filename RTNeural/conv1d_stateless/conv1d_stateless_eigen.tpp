#include "conv1d_stateless_eigen.h"

namespace RTNeural
{
template <typename T>
Conv1DStateless<T>::Conv1DStateless(int in_num_filters_in, int in_num_features_in, int in_num_filters_out, int in_kernel_size, int in_stride, bool in_valid_pad)
    : num_filters_in(in_num_filters_in)
    , num_features_in(in_num_features_in)
    , num_filters_out(in_num_filters_out)
    , kernel_size(in_kernel_size)
    , stride(in_stride)
    , valid_pad(in_valid_pad)
    , num_features_out(computeNumFeaturesOut(in_num_features_in, in_kernel_size, in_stride, in_valid_pad))
    , Layer<T>(in_num_filters_in * in_num_features_in, in_num_filters_out * computeNumFeaturesOut(in_num_features_in, in_kernel_size, in_stride, in_valid_pad))
{
    for(int i = 0; i < num_filters_out; ++i)
        kernelWeights.push_back(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(num_filters_in, kernel_size));

    // Same padding rule. Based on tensorflow: https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
    if(!valid_pad)
    {
        int total_pad = std::max(num_features_in % stride == 0 ? kernel_size - stride : kernel_size - num_features_in % stride, 0);
        pad_left = total_pad / 2;
        pad_right = total_pad - pad_left;
    }
    else
    {
        pad_left = 0;
        pad_right = 0;
    }
}

template <typename T>
Conv1DStateless<T>::Conv1DStateless(std::initializer_list<int> sizes)
    : Conv1DStateless<T>(*sizes.begin(), *(sizes.begin() + 1), *(sizes.begin() + 2), *(sizes.begin() + 3), *(sizes.begin() + 4), *(sizes.begin() + 5))
{
}

template <typename T>
Conv1DStateless<T>::Conv1DStateless(const Conv1DStateless& other)
    : Conv1DStateless(other.num_filters_in, other.num_features_in, other.num_filters_out, other.kernel_size, other.stride, other.valid_pad)
{
}

template <typename T>
Conv1DStateless<T>& Conv1DStateless<T>::operator=(const Conv1DStateless<T>& other)
{
    return *this = Conv1DStateless<T>(other);
}

template <typename T>
void Conv1DStateless<T>::setWeights(const std::vector<std::vector<std::vector<T>>>& inWeights)
{
    for(int i = 0; i < num_filters_out; ++i)
        for(int k = 0; k < num_filters_in; ++k)
            for(int j = 0; j < kernel_size; ++j)
                kernelWeights[i](k, j) = inWeights.at(i).at(k).at(j);
}

//====================================================

template <typename T, int num_filters_in, int num_features_in, int num_filters_out, int kernel_size, int stride>
Conv1DStatelessT<T, num_filters_in, num_features_in, num_filters_out, kernel_size, stride>::Conv1DStatelessT()
    : outs(outs_internal)
{
    for(int k = 0; k < num_filters_out; ++k)
        weights[k] = weights_type::Zero();
}

template <typename T, int num_filters_in, int num_features_in, int num_filters_out, int kernel_size, int stride>
void Conv1DStatelessT<T, num_filters_in, num_features_in, num_filters_out, kernel_size, stride>::setWeights(const std::vector<std::vector<std::vector<T>>>& inWeights)
{
    for(int i = 0; i < num_filters_out; ++i)
        for(int k = 0; k < num_filters_in; ++k)
            for(int j = 0; j < kernel_size; ++j)
                weights[i](k, j) = inWeights.at(i).at(k).at(j);
}
} // RTNeural
