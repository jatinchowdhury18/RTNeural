#include "conv1d_accelerate.h"

namespace RTNeural
{

template <typename T>
Conv1D<T>::Conv1D(int in_size, int out_size, int kernel_size, int dilation)
    : Layer<T>(in_size, out_size)
    , dilation_rate(dilation)
    , kernel_size(kernel_size)
    , state_size(kernel_size * dilation)
{
    kernelWeights = new T**[out_size];
    for(int i = 0; i < out_size; ++i)
    {
        kernelWeights[i] = new T*[in_size];
        for(int k = 0; k < in_size; ++k)
        {
            kernelWeights[i][k] = new T[state_size];
            std::fill(kernelWeights[i][k], &kernelWeights[i][k][state_size], (T)0);
        }
    }

    bias = new T[out_size];

    state = new T*[in_size];
    for(int k = 0; k < in_size; ++k)
        state[k] = new T[2 * state_size];
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
    for(int i = 0; i < Layer<T>::out_size; ++i)
    {
        for(int k = 0; k < Layer<T>::in_size; ++k)
            delete[] kernelWeights[i][k];

        delete[] kernelWeights[i];
    }

    delete[] kernelWeights;
    delete[] bias;

    for(int k = 0; k < Layer<T>::in_size; ++k)
        delete[] state[k];
    delete[] state;
}

template <typename T>
void Conv1D<T>::reset()
{
    state_ptr = 0;
    for(int k = 0; k < Layer<T>::in_size; ++k)
        std::fill(state[k], &state[k][2 * state_size], (T)0);
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

} // namespace RTNeural
