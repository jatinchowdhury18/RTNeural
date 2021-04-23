#ifndef CONV1D_H_INCLUDED
#define CONV1D_H_INCLUDED

#if defined(USE_EIGEN)
#include "conv1d_eigen.h"
#include "conv1d_eigen.tpp"
#elif defined(USE_XSIMD)
#include "conv1d_xsimd.h"
#include "conv1d_xsimd.tpp"
#elif defined(USE_ACCELERATE)
#include "conv1d_accelerate.h"
#include "conv1d_accelerate.tpp"
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
    Conv1D(std::initializer_list<size_t> sizes);
    Conv1D(const Conv1D& other);
    Conv1D& operator=(const Conv1D& other);
    virtual ~Conv1D();

    void reset() override;

    std::string getName() const noexcept override { return "conv1d"; }

    virtual inline void forward(const T* input, T* h) override
    {
        for(size_t k = 0; k < Layer<T>::in_size; ++k)
        {
            state[k][state_ptr] = input[k];
            state[k][state_ptr + state_size] = input[k];
        }

        for(size_t i = 0; i < Layer<T>::out_size; ++i)
        {
            h[i] = (T)0;
            for(size_t k = 0; k < Layer<T>::in_size; ++k)
                h[i] += vMult(&state[k][state_ptr], kernelWeights[i][k], state_size);

            h[i] += bias[i];
        }

        state_ptr = (state_ptr == 0 ? state_size - 1 : state_ptr - 1); // iterate state pointer in reverse
    }

    void setWeights(const std::vector<std::vector<std::vector<T>>>& weights);
    void setBias(const std::vector<T>& biasVals);

    const T getWeight(size_t outIndex, size_t inIndex, size_t kernelIndex)
    {
        return kernelWeights[outIndex][inIndex][kernelIndex];
    }

    size_t getKernelSize() const noexcept { return kernel_size; }
    size_t getDilationRate() const noexcept { return dilation_rate; }

private:
    const size_t dilation_rate;
    const size_t kernel_size;
    const size_t state_size;

    T*** kernelWeights;
    T* bias;
    T** state;
    size_t state_ptr = 0;
};

} // namespace RTNeural

#endif

#endif // CONV1D_H_INCLUDED
