#include "lstm_xsimd.h"

namespace RTNeural
{

template <typename T>
LSTMLayer<T>::LSTMLayer(size_t in_size, size_t out_size)
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
LSTMLayer<T>::LSTMLayer(std::initializer_list<size_t> sizes)
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
LSTMLayer<T>::WeightSet::WeightSet(size_t in_size, size_t out_size)
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
    for(size_t i = 0; i < Layer<T>::in_size; ++i)
    {
        for(size_t k = 0; k < Layer<T>::out_size; ++k)
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
    for(size_t i = 0; i < Layer<T>::out_size; ++i)
    {
        for(size_t k = 0; k < Layer<T>::out_size; ++k)
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
    for(size_t k = 0; k < Layer<T>::out_size; ++k)
    {
        iWeights.b[k] = bVals[k];
        fWeights.b[k] = bVals[k + Layer<T>::out_size];
        cWeights.b[k] = bVals[k + Layer<T>::out_size * 2];
        oWeights.b[k] = bVals[k + Layer<T>::out_size * 3];
    }
}

} // namespace RTNeural
