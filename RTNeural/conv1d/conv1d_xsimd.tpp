#include "conv1d_xsimd.h"

namespace RTNeural
{

template <typename T>
Conv1D<T>::Conv1D(size_t in_size, size_t out_size, size_t kernel_size, size_t dilation)
    : Layer<T>(in_size, out_size)
    , dilation_rate(dilation)
    , kernel_size(kernel_size)
    , state_size(kernel_size * dilation)
{
    kernelWeights = vec3_type(out_size, vec2_type(in_size, vec_type(state_size, (T)0)));
    bias.resize(out_size, (T)0);
    state = vec2_type(in_size, vec_type(2 * state_size, (T)0));
    prod_state.resize(state_size, (T)0);
}

template <typename T>
Conv1D<T>::Conv1D(std::initializer_list<size_t> sizes)
    : Conv1D<T>(*sizes.begin(), *(sizes.begin() + 1), *(sizes.begin() + 2), *(sizes.begin() + 3))
{
}

template <typename T>
Conv1D<T>::Conv1D(const Conv1D<T>& other)
    : Conv1D<T>(other.in_size, other.out_size, other.kernel_size, other.dilation_rate)
{
}

template <typename T>
Conv1D<T>& Conv1D<T>::operator=(const Conv1D<T>& other)
{
    return *this = Conv1D<T>(other);
}

template <typename T>
Conv1D<T>::~Conv1D()
{
}

template <typename T>
void Conv1D<T>::reset()
{
    state_ptr = 0;
    for(size_t k = 0; k < Layer<T>::in_size; ++k)
        std::fill(state[k].begin(), state[k].end(), (T)0);
}

template <typename T>
void Conv1D<T>::setWeights(const std::vector<std::vector<std::vector<T>>>& weights)
{
    for(size_t i = 0; i < Layer<T>::out_size; ++i)
        for(size_t k = 0; k < Layer<T>::in_size; ++k)
            for(size_t j = 0; j < kernel_size; ++j)
                kernelWeights[i][k][j * dilation_rate] = weights[i][k][j];
}

template <typename T>
void Conv1D<T>::setBias(const std::vector<T>& biasVals)
{
    for(size_t i = 0; i < Layer<T>::out_size; ++i)
        bias[i] = biasVals[i];
}

//====================================================
template <typename T, size_t in_sizet, size_t out_sizet, size_t kernel_size, size_t dilation_rate>
Conv1DT<T, in_sizet, out_sizet, kernel_size, dilation_rate>::Conv1DT()
{

}

template <typename T, size_t in_sizet, size_t out_sizet, size_t kernel_size, size_t dilation_rate>
void Conv1DT<T, in_sizet, out_sizet, kernel_size, dilation_rate>::reset()
{

}

template <typename T, size_t in_sizet, size_t out_sizet, size_t kernel_size, size_t dilation_rate>
void Conv1DT<T, in_sizet, out_sizet, kernel_size, dilation_rate>::setWeights(const std::vector<std::vector<std::vector<T>>>& weights)
{

}

template <typename T, size_t in_sizet, size_t out_sizet, size_t kernel_size, size_t dilation_rate>
void Conv1DT<T, in_sizet, out_sizet, kernel_size, dilation_rate>::setBias(const std::vector<T>& biasVals)
{

}

} // namespace RTNeural
