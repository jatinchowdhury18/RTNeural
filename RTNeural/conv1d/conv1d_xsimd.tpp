#include "conv1d_xsimd.h"

namespace RTNeural
{

template <typename T>
Conv1D<T>::Conv1D(int in_size, int out_size, int kernel_size, int dilation)
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
Conv1D<T>::Conv1D(std::initializer_list<int> sizes)
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
    for(int k = 0; k < Layer<T>::in_size; ++k)
        std::fill(state[k].begin(), state[k].end(), (T)0);
}

template <typename T>
void Conv1D<T>::setWeights(const std::vector<std::vector<std::vector<T>>>& weights)
{
    for(int i = 0; i < Layer<T>::out_size; ++i)
        for(int k = 0; k < Layer<T>::in_size; ++k)
            for(int j = 0; j < kernel_size; ++j)
                kernelWeights[i][k][j * dilation_rate] = weights[i][k][j];
}

template <typename T>
void Conv1D<T>::setBias(const std::vector<T>& biasVals)
{
    for(int i = 0; i < Layer<T>::out_size; ++i)
        bias[i] = biasVals[i];
}

//====================================================
template <typename T, int in_sizet, int out_sizet, int kernel_size, int dilation_rate>
Conv1DT<T, in_sizet, out_sizet, kernel_size, dilation_rate>::Conv1DT()
{
    for(int i = 0; i < out_size; ++i)
        for(int j = 0; j < v_in_size; ++j)
            for(int k = 0; k < state_size; ++k)
                weights[i][j][k] = v_type((T)0.0);

    for(int i = 0; i < v_out_size; ++i)
        bias[i] = v_type((T)0.0);

    for(int i = 0; i < v_out_size; ++i)
        outs[i] = v_type((T)0.0);

    reset();
}

template <typename T, int in_sizet, int out_sizet, int kernel_size, int dilation_rate>
void Conv1DT<T, in_sizet, out_sizet, kernel_size, dilation_rate>::reset()
{
    state_ptr = 0;
    for(int k = 0; k < v_in_size; ++k)
        for(int i = 0; i < 2 * state_size; ++i)
            state[k][i] = v_type((T)0.0);
}

template <typename T, int in_sizet, int out_sizet, int kernel_size, int dilation_rate>
void Conv1DT<T, in_sizet, out_sizet, kernel_size, dilation_rate>::setWeights(const std::vector<std::vector<std::vector<T>>>& ws)
{
    for(int i = 0; i < out_size; ++i)
    {
        for(int k = 0; k < in_size; ++k)
        {
            for(int j = 0; j < kernel_size; ++j)
            {
                auto& w = weights[i][k / v_size][j * dilation_rate];
                w = set_value(w, k % v_size, ws[i][k][j]);
            }
        }
    }
}

template <typename T, int in_sizet, int out_sizet, int kernel_size, int dilation_rate>
void Conv1DT<T, in_sizet, out_sizet, kernel_size, dilation_rate>::setBias(const std::vector<T>& biasVals)
{
    for(int i = 0; i < out_size; ++i)
        bias[i / v_size] = set_value(bias[i / v_size], i % v_size, biasVals[i]);
}

} // namespace RTNeural
