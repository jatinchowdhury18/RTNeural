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

/** Dynamic implementation of a 1-dimensional convolution layer. */
template <typename T>
class Conv1D : public Layer<T>
{
public:
    /** Constructs a convolution layer for the given dimensions. */
    Conv1D(int in_size, int out_size, int kernel_size, int dilation);
    Conv1D(std::initializer_list<int> sizes);
    Conv1D(const Conv1D& other);
    Conv1D& operator=(const Conv1D& other);
    virtual ~Conv1D();

    /** Resets the layer state. */
    void reset() override;

    /** Returns the name of this layer. */
    std::string getName() const noexcept override { return "conv1d"; }

    /** Performs forward propagation for this layer. */
    virtual inline void forward(const T* input, T* h) override
    {
        for(int k = 0; k < Layer<T>::in_size; ++k)
        {
            state[k][state_ptr] = input[k];
            state[k][state_ptr + state_size] = input[k];
        }

        for(int i = 0; i < Layer<T>::out_size; ++i)
        {
            h[i] = (T)0;
            for(int k = 0; k < Layer<T>::in_size; ++k)
                h[i] += vMult(&state[k][state_ptr], kernelWeights[i][k], state_size);

            h[i] += bias[i];
        }

        state_ptr = (state_ptr == 0 ? state_size - 1 : state_ptr - 1); // iterate state pointer in reverse
    }

    /** Sets the layer weights. */
    void setWeights(const std::vector<std::vector<std::vector<T>>>& weights);

    /** Sets the layer biases. */
    void setBias(const std::vector<T>& biasVals);

    /** Returns the weights value for the given indices. */
    const T getWeight(int outIndex, int inIndex, int kernelIndex)
    {
        return kernelWeights[outIndex][inIndex][kernelIndex];
    }

    /** Returns the size of the convolution kernel. */
    int getKernelSize() const noexcept { return kernel_size; }

    /** Returns the convolution dilation rate. */
    int getDilationRate() const noexcept { return dilation_rate; }

private:
    const int dilation_rate;
    const int kernel_size;
    const int state_size;

    T*** kernelWeights;
    T* bias;
    T** state;
    int state_ptr = 0;
};

//====================================================
/** Static implementation of a 1-dimensional convolution layer. */
template <typename T, int in_sizet, int out_sizet, int kernel_size, int dilation_rate>
class Conv1DT
{
    static constexpr auto state_size = kernel_size * dilation_rate;

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = out_sizet;

    Conv1DT();

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "conv1d"; }

    /** Returns false since convolution is not an activation layer. */
    constexpr bool isActivation() const noexcept { return false; }

    /** Resets the layer state. */
    void reset();

    /** Performs forward propagation for this layer. */
    inline void forward(const T (&ins)[in_size])
    {
        for(int k = 0; k < in_size; ++k)
        {
            state[k][state_ptr] = ins[k];
            state[k][state_ptr + state_size] = ins[k];
        }

        for(int i = 0; i < out_size; ++i)
        {
            outs[i] = bias[i];
            for(int k = 0; k < in_size; ++k)
                outs[i] += std::inner_product(&state[k][state_ptr], &state[k][state_ptr + state_size], weights[i][k], (T)0);
        }

        state_ptr = (state_ptr == 0 ? state_size - 1 : state_ptr - 1); // iterate state pointer in reverse
    }

    /** Sets the layer weights. */
    void setWeights(const std::vector<std::vector<std::vector<T>>>& weights);

    /** Sets the layer biases. */
    void setBias(const std::vector<T>& biasVals);

    /** Returns the size of the convolution kernel. */
    int getKernelSize() const noexcept { return kernel_size; }

    /** Returns the convolution dilation rate. */
    int getDilationRate() const noexcept { return dilation_rate; }

    T outs alignas(16)[out_size];

private:
    T state alignas(16)[in_size][state_size * 2];
    int state_ptr = 0;

    T weights alignas(16)[out_size][in_size][state_size];
    T bias alignas(16)[out_size];
};

} // namespace RTNeural

#endif

#endif // CONV1D_H_INCLUDED
