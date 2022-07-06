#ifndef CONV1DACCELERATE_H_INCLUDED
#define CONV1DACCELERATE_H_INCLUDED

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
    virtual inline void forward(const T* input, T* h) noexcept override
    {
        // @TODO: vectorize this!
        for(int k = 0; k < Layer<T>::in_size; ++k)
        {
            state[k][state_ptr] = input[k];
            state[k][state_ptr + state_size] = input[k];
        }

        conv_internal(h);

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

private:
    template <typename FloatType = T>
    inline typename std::enable_if<std::is_same<FloatType, float>::value>::type
    conv_internal(float* h) noexcept
    {
        float dotpr_out;
        for(int i = 0; i < Layer<T>::out_size; ++i)
        {
            h[i] = (T)0;
            for(int k = 0; k < Layer<T>::in_size; ++k)
            {
                vDSP_dotpr(&state[k][state_ptr], 1, kernelWeights[i][k], 1, &dotpr_out, state_size);
                h[i] += dotpr_out;
            }
        }

        vDSP_vadd(h, 1, bias, 1, h, 1, Layer<T>::out_size);
    }

    template <typename FloatType = T>
    inline typename std::enable_if<std::is_same<FloatType, double>::value>::type
    conv_internal(double* h) noexcept
    {
        double dotpr_out;
        for(int i = 0; i < Layer<T>::out_size; ++i)
        {
            h[i] = (T)0;
            for(int k = 0; k < Layer<T>::in_size; ++k)
            {
                vDSP_dotprD(&state[k][state_ptr], 1, kernelWeights[i][k], 1, &dotpr_out, state_size);
                h[i] += dotpr_out;
            }
        }

        vDSP_vaddD(h, 1, bias, 1, h, 1, Layer<T>::out_size);
    }

    const int dilation_rate;
    const int kernel_size;
    const int state_size;

    T*** kernelWeights;
    T* bias;
    T** state;
    int state_ptr = 0;
};

} // namespace RTNeural

#endif // CONV1DACCELERATE_H_INCLUDED
