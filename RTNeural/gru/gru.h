#ifndef GRU_H_INCLUDED
#define GRU_H_INCLUDED

#include <algorithm>

#if defined(USE_EIGEN)
#include "gru_eigen.h"
#include "gru_eigen.tpp"
#elif defined(USE_XSIMD)
#include "gru_xsimd.h"
#include "gru_xsimd.tpp"
#elif defined(USE_ACCELERATE)
#include "gru_accelerate.h"
#include "gru_accelerate.tpp"
#else
#include "../Layer.h"
#include "../common.h"
#include <vector>

namespace RTNeural
{

template <typename T>
class GRULayer : public Layer<T>
{
public:
    GRULayer(int in_size, int out_size);
    GRULayer(std::initializer_list<int> sizes);
    GRULayer(const GRULayer& other);
    GRULayer& operator=(const GRULayer& other);
    virtual ~GRULayer();

    void reset() override { std::fill(ht1, ht1 + Layer<T>::out_size, (T)0); }

    std::string getName() const noexcept override { return "gru"; }

    virtual inline void forward(const T* input, T* h) override
    {
        for(int i = 0; i < Layer<T>::out_size; ++i)
        {
            zVec[i] = sigmoid(vMult(zWeights.W[i], input, Layer<T>::in_size) + vMult(zWeights.U[i], ht1, Layer<T>::out_size) + zWeights.b[0][i] + zWeights.b[1][i]);
            rVec[i] = sigmoid(vMult(rWeights.W[i], input, Layer<T>::in_size) + vMult(rWeights.U[i], ht1, Layer<T>::out_size) + rWeights.b[0][i] + rWeights.b[1][i]);
            cVec[i] = std::tanh(vMult(cWeights.W[i], input, Layer<T>::in_size) + rVec[i] * (vMult(cWeights.U[i], ht1, Layer<T>::out_size) + cWeights.b[1][i]) + cWeights.b[0][i]);
            h[i] = ((T)1 - zVec[i]) * cVec[i] + zVec[i] * ht1[i];
        }

        std::copy(h, h + Layer<T>::out_size, ht1);
    }

    void setWVals(T** wVals);
    void setUVals(T** uVals);
    void setBVals(T** bVals);

    void setWVals(const std::vector<std::vector<T>>& wVals);
    void setUVals(const std::vector<std::vector<T>>& uVals);
    void setBVals(const std::vector<std::vector<T>>& bVals);

    T getWVal(int i, int k) const noexcept;
    T getUVal(int i, int k) const noexcept;
    T getBVal(int i, int k) const noexcept;

protected:
    T* ht1;

    struct WeightSet
    {
        WeightSet(int in_size, int out_size);
        ~WeightSet();

        T** W;
        T** U;
        T* b[2];
        const int out_size;
    };

    WeightSet zWeights;
    WeightSet rWeights;
    WeightSet cWeights;

    T* zVec;
    T* rVec;
    T* cVec;
};

//====================================================
template <typename T, int in_sizet, int out_sizet>
class GRULayerT
{
public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = out_sizet;

    GRULayerT();

    std::string getName() const noexcept { return "gru"; }
    constexpr bool isActivation() const noexcept { return false; }

    void reset();

    template <int N = in_size>
    inline typename std::enable_if<(N > 1), void>::type
    forward(const T (&ins)[in_size])
    {
        // compute zt
        recurrent_mat_mul(outs, Uz, zt);
        kernel_mat_mul(ins, Wz, kernel_outs);
        for(int i = 0; i < out_size; ++i)
            zt[i] = sigmoid(zt[i] + bz[i] + kernel_outs[i]);

        // compute rt
        recurrent_mat_mul(outs, Ur, rt);
        kernel_mat_mul(ins, Wr, kernel_outs);
        for(int i = 0; i < out_size; ++i)
            rt[i] = sigmoid(rt[i] + br[i] + kernel_outs[i]);

        // compute h_hat
        recurrent_mat_mul(outs, Uh, ct);
        kernel_mat_mul(ins, Wh, kernel_outs);
        for(int i = 0; i < out_size; ++i)
            ht[i] = std::tanh(rt[i] * (ct[i] + bh1[i]) + bh0[i] + kernel_outs[i]);

        // compute output
        for(int i = 0; i < out_size; ++i)
            outs[i] = ((T)1.0 - zt[i]) * ht[i] + zt[i] * outs[i];
    }

    template <int N = in_size>
    inline typename std::enable_if<N == 1, void>::type
    forward(const T (&ins)[in_size])
    {
        // compute zt
        recurrent_mat_mul(outs, Uz, zt);
        for(int i = 0; i < out_size; ++i)
            zt[i] = sigmoid(zt[i] + bz[i] + (Wz_1[i] * ins[0]));

        // compute rt
        recurrent_mat_mul(outs, Ur, rt);
        for(int i = 0; i < out_size; ++i)
            rt[i] = sigmoid(rt[i] + br[i] + (Wr_1[i] * ins[0]));

        // compute h_hat
        recurrent_mat_mul(outs, Uh, ct);
        for(int i = 0; i < out_size; ++i)
            ht[i] = std::tanh(rt[i] * (ct[i] + bh1[i]) + bh0[i] + (Wh_1[i] * ins[0]));

        // compute output
        for(int i = 0; i < out_size; ++i)
            outs[i] = ((T)1.0 - zt[i]) * ht[i] + zt[i] * outs[i];
    }

    void setWVals(const std::vector<std::vector<T>>& wVals);
    void setUVals(const std::vector<std::vector<T>>& uVals);
    void setBVals(const std::vector<std::vector<T>>& bVals);

    T outs alignas(16)[out_size];

private:
    static inline void recurrent_mat_mul(const T (&vec)[out_size], const T (&mat)[out_size][out_size], T (&out)[out_size]) noexcept
    {
        for(int j = 0; j < out_size; ++j)
            out[j] = std::inner_product(mat[j], mat[j] + out_size, vec, (T)0);
    }

    static inline void kernel_mat_mul(const T (&vec)[in_size], const T (&mat)[out_size][in_size], T (&out)[out_size]) noexcept
    {
        for(int j = 0; j < out_size; ++j)
            out[j] = std::inner_product(mat[j], mat[j] + in_size, vec, (T)0);
    }

    // kernel weights
    T Wr alignas(16)[out_size][in_size];
    T Wz alignas(16)[out_size][in_size];
    T Wh alignas(16)[out_size][in_size];
    T kernel_outs alignas(16)[out_size];

    // single-input kernel weights
    T Wz_1 alignas(16)[out_size];
    T Wr_1 alignas(16)[out_size];
    T Wh_1 alignas(16)[out_size];

    // recurrent weights
    T Uz alignas(16)[out_size][out_size];
    T Ur alignas(16)[out_size][out_size];
    T Uh alignas(16)[out_size][out_size];

    // biases
    T bz alignas(16)[out_size];
    T br alignas(16)[out_size];
    T bh0 alignas(16)[out_size];
    T bh1 alignas(16)[out_size];

    // intermediate vars
    T zt alignas(16)[out_size];
    T rt alignas(16)[out_size];
    T ct alignas(16)[out_size];
    T ht alignas(16)[out_size];
};

} // namespace RTNeural

#endif // USE_EIGEN

#endif // GRU_H_INCLUDED
