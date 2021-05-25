#ifndef LSTM_XSIMD_H_INCLUDED
#define LSTM_XSIMD_H_INCLUDED

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
            fVec[i] = vMult(fWeights.W[i].data(), input, prod_in.data(), Layer<T>::in_size) + vMult(fWeights.U[i].data(), ht1.data(), prod_out.data(), Layer<T>::out_size);
            iVec[i] = vMult(iWeights.W[i].data(), input, prod_in.data(), Layer<T>::in_size) + vMult(iWeights.U[i].data(), ht1.data(), prod_out.data(), Layer<T>::out_size);
            oVec[i] = vMult(oWeights.W[i].data(), input, prod_in.data(), Layer<T>::in_size) + vMult(oWeights.U[i].data(), ht1.data(), prod_out.data(), Layer<T>::out_size);
            ctVec[i] = vMult(cWeights.W[i].data(), input, prod_in.data(), Layer<T>::in_size) + vMult(cWeights.U[i].data(), ht1.data(), prod_out.data(), Layer<T>::out_size);
        }

        vAdd(fVec.data(), fWeights.b.data(), fVec.data(), Layer<T>::out_size);
        sigmoid(fVec.data(), fVec.data(), Layer<T>::out_size);

        vAdd(iVec.data(), iWeights.b.data(), iVec.data(), Layer<T>::out_size);
        sigmoid(iVec.data(), iVec.data(), Layer<T>::out_size);

        vAdd(oVec.data(), oWeights.b.data(), oVec.data(), Layer<T>::out_size);
        sigmoid(oVec.data(), oVec.data(), Layer<T>::out_size);

        vAdd(ctVec.data(), cWeights.b.data(), ctVec.data(), Layer<T>::out_size);
        tanh(ctVec.data(), ctVec.data(), Layer<T>::out_size);

        vProd(fVec.data(), ct1.data(), cVec.data(), Layer<T>::out_size);
        vProd(iVec.data(), ctVec.data(), prod_out.data(), Layer<T>::out_size);
        vAdd(cVec.data(), prod_out.data(), cVec.data(), Layer<T>::out_size);

        tanh(cVec.data(), h, Layer<T>::out_size);
        vProd(h, oVec.data(), h, Layer<T>::out_size);

        vCopy(cVec.data(), ct1.data(), Layer<T>::out_size);
        vCopy(h, ht1.data(), Layer<T>::out_size);
    }

    void setWVals(const std::vector<std::vector<T>>& wVals);
    void setUVals(const std::vector<std::vector<T>>& uVals);
    void setBVals(const std::vector<T>& bVals);

protected:
    using vec_type = std::vector<T, XSIMD_DEFAULT_ALLOCATOR(T)>;
    using vec2_type = std::vector<vec_type>;

    vec_type ht1;
    vec_type ct1;

    struct WeightSet
    {
        WeightSet(int in_size, int out_size);
        ~WeightSet();

        vec2_type W;
        vec2_type U;
        vec_type b;
        const int out_size;
    };

    WeightSet fWeights;
    WeightSet iWeights;
    WeightSet oWeights;
    WeightSet cWeights;

    vec_type fVec;
    vec_type iVec;
    vec_type oVec;
    vec_type ctVec;
    vec_type cVec;

    vec_type prod_in;
    vec_type prod_out;
};

//====================================================
template <typename T, int in_sizet, int out_sizet>
class LSTMLayerT
{
    using v_type = xsimd::simd_type<T>;
    static constexpr auto v_size = (int)v_type::size;
    static constexpr auto v_in_size = ceil_div(in_sizet, v_size);
    static constexpr auto v_out_size = ceil_div(out_sizet, v_size);

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = out_sizet;

    LSTMLayerT();

    std::string getName() const noexcept { return "lstm"; }
    constexpr bool isActivation() const noexcept { return false; }

    void reset();

    template <int N = in_size>
    inline typename std::enable_if<(N > 1), void>::type
    forward(const v_type (&ins)[v_in_size])
    {
        // compute ft
        recurrent_mat_mul(outs, Uf, ft);
        kernel_mat_mul(ins, Wf, kernel_outs);
        for(int i = 0; i < v_out_size; ++i)
            ft[i] = sigmoid(ft[i] + bf[i] + kernel_outs[i]);

        // compute it
        recurrent_mat_mul(outs, Ui, it);
        kernel_mat_mul(ins, Wi, kernel_outs);
        for(int i = 0; i < v_out_size; ++i)
            it[i] = sigmoid(it[i] + bi[i] + kernel_outs[i]);

        // compute ot
        recurrent_mat_mul(outs, Uo, ot);
        kernel_mat_mul(ins, Wo, kernel_outs);
        for(int i = 0; i < v_out_size; ++i)
            ot[i] = sigmoid(ot[i] + bo[i] + kernel_outs[i]);

        // compute ct
        recurrent_mat_mul(outs, Uc, ht);
        kernel_mat_mul(ins, Wc, kernel_outs);
        for(int i = 0; i < v_out_size; ++i)
            ct[i] = it[i] * xsimd::tanh(ht[i] + bc[i] + kernel_outs[i]) + ft[i] * ct[i];

        // compute output
        for(int i = 0; i < v_out_size; ++i)
            outs[i] = ot[i] * xsimd::tanh(ct[i]);
    }

    template <int N = in_size>
    inline typename std::enable_if<N == 1, void>::type
    forward(const v_type (&ins)[v_in_size])
    {
        // compute ft
        recurrent_mat_mul(outs, Uf, ft);
        for(int i = 0; i < v_out_size; ++i)
            ft[i] = sigmoid(ft[i] + bf[i] + (Wf_1[i] * ins[0]));

        // compute it
        recurrent_mat_mul(outs, Ui, it);
        for(int i = 0; i < v_out_size; ++i)
            it[i] = sigmoid(it[i] + bi[i] + (Wi_1[i] * ins[0]));

        // compute ot
        recurrent_mat_mul(outs, Uo, ot);
        for(int i = 0; i < v_out_size; ++i)
            ot[i] = sigmoid(ot[i] + bo[i] + (Wo_1[i] * ins[0]));

        // compute ct
        recurrent_mat_mul(outs, Uc, ht);
        for(int i = 0; i < v_out_size; ++i)
            ct[i] = it[i] * xsimd::tanh(ht[i] + bc[i] + (Wc_1[i] * ins[0])) + ft[i] * ct[i];

        // compute output
        for(int i = 0; i < v_out_size; ++i)
            outs[i] = ot[i] * xsimd::tanh(ct[i]);
    }

    void setWVals(const std::vector<std::vector<T>>& wVals);
    void setUVals(const std::vector<std::vector<T>>& uVals);
    void setBVals(const std::vector<T>& bVals);

    v_type outs[v_out_size];

private:
    static inline void recurrent_mat_mul(const v_type (&vec)[v_out_size], const v_type (&mat)[out_size][v_out_size], v_type (&out)[v_out_size]) noexcept
    {
        T sums alignas(16)[out_size] { (T)0 };
        for(int i = 0; i < v_size; ++i)
        {
            for(int j = 0; j < v_out_size; ++j)
            {
                for(int k = 0; k < v_out_size; ++k)
                    sums[i + j * v_size] += xsimd::hadd(mat[i + j * v_size][k] * vec[k]);
            }
        }

        for(int i = 0; i < v_out_size; ++i)
            out[i] = xsimd::load_aligned(sums + i * v_size);
    }

    static inline void kernel_mat_mul(const v_type (&vec)[v_in_size], const v_type (&mat)[out_size][v_in_size], v_type (&out)[v_out_size]) noexcept
    {
        T sums alignas(16)[out_size] { (T)0 };
        for(int i = 0; i < v_size; ++i)
        {
            for(int j = 0; j < v_out_size; ++j)
            {
                for(int k = 0; k < v_in_size; ++k)
                    sums[i + j * v_size] += xsimd::hadd(mat[i + j * v_size][k] * vec[k]);
            }
        }

        for(int i = 0; i < v_out_size; ++i)
            out[i] = xsimd::load_aligned(sums + i * v_size);
    }

    static inline v_type sigmoid(v_type x) noexcept
    {
        return (T)1.0 / ((T)1.0 + xsimd::exp(-x));
    }

    // kernel weights
    v_type Wf[out_size][v_in_size];
    v_type Wi[out_size][v_in_size];
    v_type Wo[out_size][v_in_size];
    v_type Wc[out_size][v_in_size];
    v_type kernel_outs[v_out_size];

    // single-input kernel weights
    v_type Wf_1[v_out_size];
    v_type Wi_1[v_out_size];
    v_type Wo_1[v_out_size];
    v_type Wc_1[v_out_size];

    // recurrent weights
    v_type Uf[out_size][v_out_size];
    v_type Ui[out_size][v_out_size];
    v_type Uo[out_size][v_out_size];
    v_type Uc[out_size][v_out_size];

    // biases
    v_type bf[v_out_size];
    v_type bi[v_out_size];
    v_type bo[v_out_size];
    v_type bc[v_out_size];

    // intermediate vars
    v_type ft[v_out_size];
    v_type it[v_out_size];
    v_type ot[v_out_size];
    v_type ht[v_out_size];
    v_type ct[v_out_size];
};

} // namespace RTNeural

#endif // LSTM_XSIMD_H_INCLUDED
