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

    void reset() override;

    virtual inline void forward(const T* input, T* h) override
    {
        // copy input into state vector
        std::copy(&state[Layer<T>::in_size], &state[state_size], state);
        std::copy(input, &input[Layer<T>::in_size], &state[state_size - Layer<T>::in_size]);

        // zero output vector
        std::fill(h, &h[Layer<T>::out_size], (T) 0);

        for(size_t i = 0; i < Layer<T>::out_size; ++i)
        {
            for(size_t k = 0; k < Layer<T>::in_size; ++k)
                h[i] += vMult(&state[k], kernelWeights[i][k], kernel_size * dilation_rate);
                
            h[i] += bias[i];
        }
    }

    void setWeights(const std::vector<std::vector<std::vector<T>>>& weights);
    void setBias(const std::vector<T>& biasVals);

private:
    const size_t dilation_rate;
    const size_t kernel_size;
    const size_t state_size;

    T*** kernelWeights;
    T* bias;
    T* state;
};

} // namespace RTNeural

#endif

#endif // CONV1D_H_INCLUDED
