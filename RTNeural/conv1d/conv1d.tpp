#include "conv1d.h"

namespace RTNeural
{

#if !defined(USE_EIGEN) && !defined(USE_XSIMD)

template <typename T>
Conv1D<T>::Conv1D(size_t in_size, size_t out_size, size_t kernel_size, size_t dilation)
    : Layer<T>(in_size, out_size)
    , dilation_rate(dilation)
    , kernel_size(kernel_size)
    , state_size(std::max((out_size + kernel_size - 1) * dilation, in_size))
{
    kernelWeights = new T**[out_size];
    for(size_t i = 0; i < out_size; ++i)
    {
        kernelWeights[i] = new T*[in_size];
        for(size_t k = 0; k < in_size; ++k)
        {
            kernelWeights[i][k] = new T[kernel_size * dilation];
            std::fill(kernelWeights[i][k], &kernelWeights[i][k][kernel_size * dilation], (T) 0);
        }
    }

    bias = new T[out_size];
    state = new T[state_size];
}

template <typename T>
Conv1D<T>::~Conv1D()
{
    for(size_t i = 0; i < Layer<T>::out_size; ++i)
    {
        for(size_t k = 0; k < Layer<T>::in_size; ++k)
            delete[] kernelWeights[i][k];

        delete[] kernelWeights[i];
    }

    delete[] kernelWeights;
    delete[] bias;
    delete[] state;
}

template <typename T>
void Conv1D<T>::reset()
{
    std::fill(state, &state[state_size], (T) 0);
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

#endif

} // namespace RTNeural
