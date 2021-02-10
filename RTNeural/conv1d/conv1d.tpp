#include "conv1d.h"

namespace RTNeural
{

#if !defined(USE_EIGEN) && !defined(USE_XSIMD)

template <typename T>
Conv1D<T>::Conv1D(size_t in_size, size_t out_size, size_t kernel_size, size_t dilation)
    : Layer<T>(in_size, out_size)
    , dilation_rate(dilation)
    , full_kernel_size(kernel_size * dilation)
{
    kernelWeights = new T*[out_size];
    for(size_t i = 0; i < out_size; ++i)
    {
        kernelWeights[i] = new T[full_kernel_size];
        std::fill(kernelWeights[i], &kernelWeights[i][full_kernerl_size], (T) 0);
    }

    bias = new T[out_size];
}

template <typename T>
Conv1D<T>::~Conv1D()
{
    for(size_t i = 0; i < Layer<T>::out_size; ++i)
        delete[] kernelWeights[i];

    delete[] kernelWeights;
    delete[] bias;
}

#endif

} // namespace RTNeural
