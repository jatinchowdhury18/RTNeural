#ifndef CONV1DACCELERATE_H_INCLUDED
#define CONV1DACCELERATE_H_INCLUDED

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

    virtual inline void forward(const T* input, T* h) override
    {
        // @TODO: vectorize this!
        for(size_t k = 0; k < Layer<T>::in_size; ++k)
        {
            state[k][state_ptr] = input[k];
            state[k][state_ptr + state_size] = input[k];
        }

        conv_internal(h);

        state_ptr = (state_ptr == 0 ? state_size - 1 : state_ptr - 1); // iterate state pointer in reverse
    }

    void setWeights(const std::vector<std::vector<std::vector<T>>>& weights);
    void setBias(const std::vector<T>& biasVals);

    size_t getKernelSize() const noexcept { return kernel_size; }

private:
    template <typename FloatType = T>
    inline typename std::enable_if<std::is_same<FloatType, float>::value>::type
    conv_internal(float* h)
    {
        float dotpr_out;
        for(size_t i = 0; i < Layer<T>::out_size; ++i)
        {
            h[i] = (T)0;
            for(size_t k = 0; k < Layer<T>::in_size; ++k)
            {
                vDSP_dotpr(&state[k][state_ptr], 1, kernelWeights[i][k], 1, &dotpr_out, state_size);
                h[i] += dotpr_out;
            }
        }

        vDSP_vadd(h, 1, bias, 1, h, 1, Layer<T>::out_size);
    }

    template <typename FloatType = T>
    inline typename std::enable_if<std::is_same<FloatType, double>::value>::type
    conv_internal(double* h)
    {
        double dotpr_out;
        for(size_t i = 0; i < Layer<T>::out_size; ++i)
        {
            h[i] = (T)0;
            for(size_t k = 0; k < Layer<T>::in_size; ++k)
            {
                vDSP_dotprD(&state[k][state_ptr], 1, kernelWeights[i][k], 1, &dotpr_out, state_size);
                h[i] += dotpr_out;
            }
        }

        vDSP_vaddD(h, 1, bias, 1, h, 1, Layer<T>::out_size);
    }

    const size_t dilation_rate;
    const size_t kernel_size;
    const size_t state_size;

    T*** kernelWeights;
    T* bias;
    T** state;
    size_t state_ptr = 0;
};

} // namespace RTNeural

#endif // CONV1DACCELERATE_H_INCLUDED
