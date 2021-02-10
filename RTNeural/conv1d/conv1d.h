#ifndef CONV1D_H_INCLUDED
#define CONV1D_H_INCLUDED

#if defined(USE_EIGEN)
#include "conv1d_eigen.h"
#include "conv1d_eigen.tpp"
#elif defined(USE_XSIMD)
#include "conv1d_xsimd.h"
#include "conv1d_xsimd.tpp"
#else
#include "../Layer.h"
#include "../common.h"
#include <vector>

namespace RTNeural
{

template <typename T>
class Conv1D : public Layer<T>
{
public:
    Conv1D(size_t in_size, size_t out_size, size_t kernel_size, size_t dilation);
    virtual ~Conv1D();

    virtual inline void forward(const T* input, T* h) override
    {
        for(size_t i = 0; i < Layer<T>::out_size; ++i)
            h[i] = vMult() + bias[i];
    }

private:
    const size_t dilation_rate;
    const size_t full_kernel_size;

    T** kernelWeights;
    T* bias;
};

} // namespace RTNeural

#endif // CONV1D_H_INCLUDED
