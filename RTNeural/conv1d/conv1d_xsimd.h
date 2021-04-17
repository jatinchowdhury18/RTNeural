#ifndef CONV1DXSIMD_H_INCLUDED
#define CONV1DXSIMD_H_INCLUDED

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

        for(size_t i = 0; i < Layer<T>::out_size; ++i)
        {
            h[i] = (T)0;
            for(size_t k = 0; k < Layer<T>::in_size; ++k)
                h[i] += vMult(&state[k][state_ptr], kernelWeights[i][k].data(), prod_state.data(), state_size);
        }

        vAdd(h, bias.data(), h, Layer<T>::out_size);

        state_ptr = (state_ptr == 0 ? state_size - 1 : state_ptr - 1); // iterate state pointer in reverse
    }

    void setWeights(const std::vector<std::vector<std::vector<T>>>& weights);
    void setBias(const std::vector<T>& biasVals);

    size_t getKernelSize() const noexcept { return kernel_size; }
    size_t getDilationRate() const noexcept { return dilation_rate; }

private:
    using vec_type = std::vector<T, XSIMD_DEFAULT_ALLOCATOR(T)>;
    using vec2_type = std::vector<vec_type>;
    using vec3_type = std::vector<vec2_type>;

    const size_t dilation_rate;
    const size_t kernel_size;
    const size_t state_size;

    vec3_type kernelWeights;
    // T*** kernelWeights;
    vec_type bias;
    vec2_type state;
    size_t state_ptr = 0;

    vec_type prod_state;
};

} // namespace RTNeural

#endif // CONV1DXSIMD_H_INCLUDED
