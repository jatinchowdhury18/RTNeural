#ifndef LSTM_H_INCLUDED
#define LSTM_H_INCLUDED

#if defined(USE_EIGEN)
#include "lstm_eigen.h"
#include "lstm_eigen.tpp"
#elif defined(USE_XSIMD)
#include "lstm_xsimd.h"
#include "lstm_xsimd.tpp"
#elif defined(USE_ACCELERATE)
#include "lstm_accelerate.h"
#include "lstm_accelerate.tpp"
#else
#include "../Layer.h"
#include "../common.h"
#include <vector>

namespace RTNeural
{

template <typename T>
class LSTMLayer : public Layer<T>
{
public:
    LSTMLayer(int in_size, int out_size);
    LSTMLayer(std::initializer_list<int> sizes);
    LSTMLayer(const LSTMLayer& other);
    LSTMLayer& operator=(const LSTMLayer& other);
    virtual ~LSTMLayer();

    void reset() override;

    std::string getName() const noexcept override { return "lstm"; }

    virtual inline void forward(const T* input, T* h) override
    {
        for(int i = 0; i < Layer<T>::out_size; ++i)
        {
            fVec[i] = sigmoid(vMult(fWeights.W[i], input, Layer<T>::in_size) + vMult(fWeights.U[i], ht1, Layer<T>::out_size) + fWeights.b[i]);
            iVec[i] = sigmoid(vMult(iWeights.W[i], input, Layer<T>::in_size) + vMult(iWeights.U[i], ht1, Layer<T>::out_size) + iWeights.b[i]);
            oVec[i] = sigmoid(vMult(oWeights.W[i], input, Layer<T>::in_size) + vMult(oWeights.U[i], ht1, Layer<T>::out_size) + oWeights.b[i]);
            ctVec[i] = std::tanh(vMult(cWeights.W[i], input, Layer<T>::in_size) + vMult(cWeights.U[i], ht1, Layer<T>::out_size) + cWeights.b[i]);
            cVec[i] = fVec[i] * ct1[i] + iVec[i] * ctVec[i];
            h[i] = oVec[i] * std::tanh(cVec[i]);
        }

        std::copy(cVec, cVec + Layer<T>::out_size, ct1);
        std::copy(h, h + Layer<T>::out_size, ht1);
    }

    void setWVals(const std::vector<std::vector<T>>& wVals);
    void setUVals(const std::vector<std::vector<T>>& uVals);
    void setBVals(const std::vector<T>& bVals);

protected:
    T* ht1;
    T* ct1;

    /** Struct to hold layer weights (used internally) */
    struct WeightSet
    {
        WeightSet(int in_size, int out_size);
        ~WeightSet();

        T** W;
        T** U;
        T* b;
        const int out_size;
    };

    WeightSet fWeights;
    WeightSet iWeights;
    WeightSet oWeights;
    WeightSet cWeights;

    T* fVec;
    T* iVec;
    T* oVec;
    T* ctVec;
    T* cVec;
};

//====================================================
template <typename T, int in_sizet, int out_sizet>
class LSTMLayerT
{
public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = out_sizet;

    LSTMLayerT();

    std::string getName() const noexcept { return "lstm"; }
    constexpr bool isActivation() const noexcept { return false; }

    void reset();

    template <int N = in_size>
    inline typename std::enable_if<(N > 1), void>::type
    forward(const T (&ins)[in_size])
    {
        // compute ft
        recurrent_mat_mul(outs, Uf, ft);
        kernel_mat_mul(ins, Wf, kernel_outs);
        for(int i = 0; i < out_size; ++i)
            ft[i] = sigmoid(ft[i] + bf[i] + kernel_outs[i]);

        // compute it
        recurrent_mat_mul(outs, Ui, it);
        kernel_mat_mul(ins, Wi, kernel_outs);
        for(int i = 0; i < out_size; ++i)
            it[i] = sigmoid(it[i] + bi[i] + kernel_outs[i]);

        // compute ot
        recurrent_mat_mul(outs, Uo, ot);
        kernel_mat_mul(ins, Wo, kernel_outs);
        for(int i = 0; i < out_size; ++i)
            ot[i] = sigmoid(ot[i] + bo[i] + kernel_outs[i]);

        // compute ct
        recurrent_mat_mul(outs, Uc, ht);
        kernel_mat_mul(ins, Wc, kernel_outs);
        for(int i = 0; i < out_size; ++i)
            ct[i] = it[i] * std::tanh(ht[i] + bc[i] + kernel_outs[i]) + ft[i] * ct[i];

        // compute output
        for(int i = 0; i < out_size; ++i)
            outs[i] = ot[i] * std::tanh(ct[i]);
    }

    template <int N = in_size>
    inline typename std::enable_if<N == 1, void>::type
    forward(const T (&ins)[in_size])
    {
        // compute ft
        recurrent_mat_mul(outs, Uf, ft);
        for(int i = 0; i < out_size; ++i)
            ft[i] = sigmoid(ft[i] + bf[i] + (Wf_1[i] * ins[0]));

        // compute it
        recurrent_mat_mul(outs, Ui, it);
        for(int i = 0; i < out_size; ++i)
            it[i] = sigmoid(it[i] + bi[i] + (Wi_1[i] * ins[0]));

        // compute ot
        recurrent_mat_mul(outs, Uo, ot);
        for(int i = 0; i < out_size; ++i)
            ot[i] = sigmoid(ot[i] + bo[i] + (Wo_1[i] * ins[0]));

        // compute ct
        recurrent_mat_mul(outs, Uc, ht);
        for(int i = 0; i < out_size; ++i)
            ct[i] = it[i] * std::tanh(ht[i] + bc[i] + (Wc_1[i] * ins[0])) + ft[i] * ct[i];

        // compute output
        for(int i = 0; i < out_size; ++i)
            outs[i] = ot[i] * std::tanh(ct[i]);
    }

    void setWVals(const std::vector<std::vector<T>>& wVals);
    void setUVals(const std::vector<std::vector<T>>& uVals);
    void setBVals(const std::vector<T>& bVals);

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
    T Wf alignas(16)[out_size][in_size];
    T Wi alignas(16)[out_size][in_size];
    T Wo alignas(16)[out_size][in_size];
    T Wc alignas(16)[out_size][in_size];
    T kernel_outs alignas(16)[out_size];

    // single-input kernel weights
    T Wf_1 alignas(16)[out_size];
    T Wi_1 alignas(16)[out_size];
    T Wo_1 alignas(16)[out_size];
    T Wc_1 alignas(16)[out_size];

    // recurrent weights
    T Uf alignas(16)[out_size][out_size];
    T Ui alignas(16)[out_size][out_size];
    T Uo alignas(16)[out_size][out_size];
    T Uc alignas(16)[out_size][out_size];

    // biases
    T bf alignas(16)[out_size];
    T bi alignas(16)[out_size];
    T bo alignas(16)[out_size];
    T bc alignas(16)[out_size];

    // intermediate vars
    T ft alignas(16)[out_size];
    T it alignas(16)[out_size];
    T ot alignas(16)[out_size];
    T ht alignas(16)[out_size];
    T ct alignas(16)[out_size];
};

} // namespace RTNeural

#endif

#endif // LSTM_H_INCLUDED
