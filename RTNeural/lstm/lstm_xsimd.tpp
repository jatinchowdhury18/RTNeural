#include "lstm_xsimd.h"

namespace RTNeural
{

template <typename T>
LSTMLayer<T>::LSTMLayer(int in_size, int out_size)
    : Layer<T>(in_size, out_size)
    , fWeights(in_size, out_size)
    , iWeights(in_size, out_size)
    , oWeights(in_size, out_size)
    , cWeights(in_size, out_size)
{
    ht1.resize(out_size, (T)0);
    ct1.resize(out_size, (T)0);

    fVec.resize(out_size, (T)0);
    iVec.resize(out_size, (T)0);
    oVec.resize(out_size, (T)0);
    ctVec.resize(out_size, (T)0);
    cVec.resize(out_size, (T)0);

    prod_in.resize(in_size, (T)0);
    prod_out.resize(out_size, (T)0);
}

template <typename T>
LSTMLayer<T>::LSTMLayer(std::initializer_list<int> sizes)
    : LSTMLayer<T>(*sizes.begin(), *(sizes.begin() + 1))
{
}

template <typename T>
LSTMLayer<T>::LSTMLayer(const LSTMLayer& other)
    : LSTMLayer<T>(other.in_size, other.out_size)
{
}

template <typename T>
LSTMLayer<T>& LSTMLayer<T>::operator=(const LSTMLayer<T>& other)
{
    return *this = LSTMLayer<T>(other);
}

template <typename T>
LSTMLayer<T>::~LSTMLayer()
{
}

template <typename T>
void LSTMLayer<T>::reset()
{
    std::fill(ht1.begin(), ht1.end(), (T)0);
    std::fill(ct1.begin(), ct1.end(), (T)0);
}

template <typename T>
LSTMLayer<T>::WeightSet::WeightSet(int in_size, int out_size)
    : out_size(out_size)
{
    W = vec2_type(out_size, vec_type(in_size, (T)0));
    U = vec2_type(out_size, vec_type(out_size, (T)0));
    b.resize(out_size, (T)0);
}

template <typename T>
LSTMLayer<T>::WeightSet::~WeightSet()
{
}

template <typename T>
void LSTMLayer<T>::setWVals(const std::vector<std::vector<T>>& wVals)
{
    for(int i = 0; i < Layer<T>::in_size; ++i)
    {
        for(int k = 0; k < Layer<T>::out_size; ++k)
        {
            iWeights.W[k][i] = wVals[i][k];
            fWeights.W[k][i] = wVals[i][k + Layer<T>::out_size];
            cWeights.W[k][i] = wVals[i][k + Layer<T>::out_size * 2];
            oWeights.W[k][i] = wVals[i][k + Layer<T>::out_size * 3];
        }
    }
}

template <typename T>
void LSTMLayer<T>::setUVals(const std::vector<std::vector<T>>& uVals)
{
    for(int i = 0; i < Layer<T>::out_size; ++i)
    {
        for(int k = 0; k < Layer<T>::out_size; ++k)
        {
            iWeights.U[k][i] = uVals[i][k];
            fWeights.U[k][i] = uVals[i][k + Layer<T>::out_size];
            cWeights.U[k][i] = uVals[i][k + Layer<T>::out_size * 2];
            oWeights.U[k][i] = uVals[i][k + Layer<T>::out_size * 3];
        }
    }
}

template <typename T>
void LSTMLayer<T>::setBVals(const std::vector<T>& bVals)
{
    for(int k = 0; k < Layer<T>::out_size; ++k)
    {
        iWeights.b[k] = bVals[k];
        fWeights.b[k] = bVals[k + Layer<T>::out_size];
        cWeights.b[k] = bVals[k + Layer<T>::out_size * 2];
        oWeights.b[k] = bVals[k + Layer<T>::out_size * 3];
    }
}

//====================================================
template <typename T, int in_sizet, int out_sizet>
LSTMLayerT<T, in_sizet, out_sizet>::LSTMLayerT()
{
    for(int i = 0; i < v_out_size; ++i)
    {
        // single-input kernel weights
        Wf_1[i] = v_type((T)0);
        Wi_1[i] = v_type((T)0);
        Wo_1[i] = v_type((T)0);
        Wc_1[i] = v_type((T)0);

        // biases
        bf[i] = v_type((T)0);
        bi[i] = v_type((T)0);
        bo[i] = v_type((T)0);
        bc[i] = v_type((T)0);

        // intermediate vars
        ft[i] = v_type((T)0);
        it[i] = v_type((T)0);
        ot[i] = v_type((T)0);
        ht[i] = v_type((T)0);
    }

    for(int i = 0; i < out_size; ++i)
    {
        // recurrent weights
        for(int k = 0; k < v_out_size; ++k)
        {
            Uf[i][k] = v_type((T)0);
            Ui[i][k] = v_type((T)0);
            Uo[i][k] = v_type((T)0);
            Uc[i][k] = v_type((T)0);
        }

        // kernel weights
        for(int k = 0; k < v_in_size; ++k)
        {
            Wf[i][k] = v_type((T)0);
            Wi[i][k] = v_type((T)0);
            Wo[i][k] = v_type((T)0);
            Wc[i][k] = v_type((T)0);
        }
    }

    reset();
}

template <typename T, int in_sizet, int out_sizet>
void LSTMLayerT<T, in_sizet, out_sizet>::reset()
{
    // reset output state
    for(int i = 0; i < v_out_size; ++i)
    {
        ct[i] = v_type((T)0);
        outs[i] = v_type((T)0);
    }
}

template <typename T, int in_sizet, int out_sizet>
void LSTMLayerT<T, in_sizet, out_sizet>::setWVals(const std::vector<std::vector<T>>& wVals)
{
    for(int i = 0; i < in_size; ++i)
    {
        for(int j = 0; j < out_size; ++j)
        {
            Wi[j][i / v_size] = set_value(Wi[j][i / v_size], i % v_size, wVals[i][j]);
            Wf[j][i / v_size] = set_value(Wf[j][i / v_size], i % v_size, wVals[i][j + out_size]);
            Wc[j][i / v_size] = set_value(Wc[j][i / v_size], i % v_size, wVals[i][j + 2 * out_size]);
            Wo[j][i / v_size] = set_value(Wo[j][i / v_size], i % v_size, wVals[i][j + 3 * out_size]);
        }
    }

    for(int j = 0; j < out_size; ++j)
    {
        Wi_1[j / v_size] = set_value(Wi_1[j / v_size], j % v_size, wVals[0][j]);
        Wf_1[j / v_size] = set_value(Wf_1[j / v_size], j % v_size, wVals[0][j + out_size]);
        Wc_1[j / v_size] = set_value(Wc_1[j / v_size], j % v_size, wVals[0][j + 2 * out_size]);
        Wo_1[j / v_size] = set_value(Wo_1[j / v_size], j % v_size, wVals[0][j + 3 * out_size]);
    }
}

template <typename T, int in_sizet, int out_sizet>
void LSTMLayerT<T, in_sizet, out_sizet>::setUVals(const std::vector<std::vector<T>>& uVals)
{
    for(int i = 0; i < out_size; ++i)
    {
        for(int j = 0; j < out_size; ++j)
        {
            Ui[j][i / v_size] = set_value(Ui[j][i / v_size], i % v_size, uVals[i][j]);
            Uf[j][i / v_size] = set_value(Uf[j][i / v_size], i % v_size, uVals[i][j + out_size]);
            Uc[j][i / v_size] = set_value(Uc[j][i / v_size], i % v_size, uVals[i][j + 2 * out_size]);
            Uo[j][i / v_size] = set_value(Uo[j][i / v_size], i % v_size, uVals[i][j + 3 * out_size]);
        }
    }
}

template <typename T, int in_sizet, int out_sizet>
void LSTMLayerT<T, in_sizet, out_sizet>::setBVals(const std::vector<T>& bVals)
{
    for(int k = 0; k < out_size; ++k)
    {
        bi[k / v_size] = set_value(bi[k / v_size], k % v_size, bVals[k]);
        bf[k / v_size] = set_value(bf[k / v_size], k % v_size, bVals[k + out_size]);
        bc[k / v_size] = set_value(bc[k / v_size], k % v_size, bVals[k + 2 * out_size]);
        bo[k / v_size] = set_value(bo[k / v_size], k % v_size, bVals[k + 3 * out_size]);
    }
}

} // namespace RTNeural
