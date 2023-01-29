#include "conv1d_stateless_eigen.h"

namespace RTNeural
{

template <typename T, int num_filters_in, int num_features_in, int num_filters_out, int kernel_size, int stride>
Conv1DStatelessT<T, num_filters_in, num_features_in, num_filters_out, kernel_size, stride>::Conv1DStatelessT()
    : outs(outs_internal)
{
    for(int k = 0; k < num_filters_out; ++k)
        weights[k] = weights_type::Zero();

    bias = bias_type::Zero();
}

template <typename T, int num_filters_in, int num_features_in, int num_filters_out, int kernel_size, int stride>
void Conv1DStatelessT<T, num_filters_in, num_features_in, num_filters_out, kernel_size, stride>::setWeights(const std::vector<std::vector<std::vector<T>>>& inWeights)
{
    for(int i = 0; i < num_features_out; ++i)
        for(int k = 0; k < num_features_in; ++k)
            for(int j = 0; j < kernel_size; ++j)
                weights[i](k, j) = inWeights[i][k][j];
}

template <typename T, int num_filters_in, int num_features_in, int num_filters_out, int kernel_size, int stride>
void Conv1DStatelessT<T, num_filters_in, num_features_in, num_filters_out, kernel_size, stride>::setBias(const std::vector<T>& inBias)
{
    for(int i = 0; i < num_features_out; ++i)
        bias(i) = inBias[i];
}
} // RTNeural
