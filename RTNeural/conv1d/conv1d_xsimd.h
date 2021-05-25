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
    Conv1D(int in_size, int out_size, int kernel_size, int dilation);
    Conv1D(std::initializer_list<int> sizes);
    Conv1D(const Conv1D& other);
    Conv1D& operator=(const Conv1D& other);
    virtual ~Conv1D();

    void reset() override;

    virtual inline void forward(const T* input, T* h) override
    {
        // @TODO: vectorize this!
        for(int k = 0; k < Layer<T>::in_size; ++k)
        {
            state[k][state_ptr] = input[k];
            state[k][state_ptr + state_size] = input[k];
        }

        for(int i = 0; i < Layer<T>::out_size; ++i)
        {
            h[i] = (T)0;
            for(int k = 0; k < Layer<T>::in_size; ++k)
                h[i] += vMult(&state[k][state_ptr], kernelWeights[i][k].data(), prod_state.data(), state_size);
        }

        vAdd(h, bias.data(), h, Layer<T>::out_size);

        state_ptr = (state_ptr == 0 ? state_size - 1 : state_ptr - 1); // iterate state pointer in reverse
    }

    void setWeights(const std::vector<std::vector<std::vector<T>>>& weights);
    void setBias(const std::vector<T>& biasVals);

    int getKernelSize() const noexcept { return kernel_size; }
    int getDilationRate() const noexcept { return dilation_rate; }

private:
    using vec_type = std::vector<T, XSIMD_DEFAULT_ALLOCATOR(T)>;
    using vec2_type = std::vector<vec_type>;
    using vec3_type = std::vector<vec2_type>;

    const int dilation_rate;
    const int kernel_size;
    const int state_size;

    vec3_type kernelWeights;
    vec_type bias;
    vec2_type state;
    int state_ptr = 0;

    vec_type prod_state;
};

//====================================================
template <typename T, int in_sizet, int out_sizet, int kernel_size, int dilation_rate>
class Conv1DT
{
    using v_type = xsimd::simd_type<T>;
    static constexpr auto v_size = (int) v_type::size;
    static constexpr auto v_in_size = ceil_div(in_sizet, v_size);
    static constexpr auto v_out_size = ceil_div(out_sizet, v_size);
    static constexpr auto state_size = kernel_size * dilation_rate;
    static constexpr auto v_state_size = ceil_div(state_size, v_size);

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = out_sizet;

    Conv1DT();

    std::string getName() const noexcept { return "conv1d"; }
    constexpr bool isActivation() const noexcept { return false; }

    void reset();

    inline void forward(const v_type (&ins)[v_in_size])
    {
        for(int k = 0; k < v_in_size; ++k)
        {
            state[k][state_ptr] = ins[k];
            state[k][state_ptr + state_size] = ins[k];
        }

        for(int i = 0; i < v_out_size; ++i)
        {
            T out_sum alignas(16)[v_size] { (T)0 };
            for(int j = 0; j < v_in_size; ++j)
            {
                for(int k = 0; k < v_size; ++k)
                {
                    for(int l = 0; l < state_size; ++l)
                        out_sum[k] += xsimd::hadd(state[j][state_ptr + l] * weights[i * v_size + k][j][l]);
                }
            }

            outs[i] = xsimd::load_aligned(out_sum) + bias[i];
        }

        state_ptr = (state_ptr == 0 ? state_size - 1 : state_ptr - 1); // iterate state pointer in reverse
    }

    void setWeights(const std::vector<std::vector<std::vector<T>>>& weights);
    void setBias(const std::vector<T>& biasVals);

    constexpr int getKernelSize() const { return kernel_size; }
    constexpr int getDilationRate() const { return dilation_rate; }

    v_type outs[v_out_size];

private:
    v_type state[v_in_size][state_size * 2];
    int state_ptr = 0;

    v_type weights[out_size][v_in_size][state_size];
    v_type bias[v_out_size];
};

} // namespace RTNeural

#endif // CONV1DXSIMD_H_INCLUDED
