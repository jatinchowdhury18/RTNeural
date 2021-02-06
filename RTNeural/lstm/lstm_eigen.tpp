#include "lstm_eigen.h"

namespace RTNeural
{

template<typename T>
LSTMLayer<T>::LSTMLayer (size_t in_size, size_t out_size) :
    Layer<T> (in_size, out_size)
{
    Wf.resize (out_size, in_size);
    Wi.resize (out_size, in_size);
    Wo.resize (out_size, in_size);
    Wc.resize (out_size, in_size);

    Uf.resize (out_size, out_size);
    Ui.resize (out_size, out_size);
    Uo.resize (out_size, out_size);
    Uc.resize (out_size, out_size);

    bf.resize (out_size, 1);
    bi.resize (out_size, 1);
    bo.resize (out_size, 1);
    bc.resize (out_size, 1);

    fVec.resize (out_size, 1);
    iVec.resize (out_size, 1);
    oVec.resize (out_size, 1);
    ctVec.resize (out_size, 1);
    cVec.resize (out_size, 1);

    inVec.resize (out_size, 1);
    ht1.resize (out_size, 1);
    ct1.resize (out_size, 1);
}

template<typename T>
void LSTMLayer<T>::reset()
{
    std::fill(ht1.data(), ht1.data() + Layer<T>::out_size, (T) 0);
    std::fill(ct1.data(), ct1.data() + Layer<T>::out_size, (T) 0);
}

template<typename T>
void LSTMLayer<T>::setWVals(const std::vector<std::vector<T>>& wVals)
{
    for (size_t i = 0; i < Layer<T>::in_size; ++i)
    {
        for (size_t k = 0; k < Layer<T>::out_size; ++k)
        {
            Wi(k, i) = wVals[i][k];
            Wf(k, i) = wVals[i][k+Layer<T>::out_size];
            Wc(k, i) = wVals[i][k+Layer<T>::out_size*2];
            Wo(k, i) = wVals[i][k+Layer<T>::out_size*3];
        }
    }
}

template<typename T>
void LSTMLayer<T>::setUVals(const std::vector<std::vector<T>>& uVals)
{
    for (size_t i = 0; i < Layer<T>::out_size; ++i)
    {
        for (size_t k = 0; k < Layer<T>::out_size; ++k)
        {
            Ui(k, i) = uVals[i][k];
            Uf(k, i) = uVals[i][k+Layer<T>::out_size];
            Uc(k, i) = uVals[i][k+Layer<T>::out_size*2];
            Uo(k, i) = uVals[i][k+Layer<T>::out_size*3];
        }
    }
}

template<typename T>
void LSTMLayer<T>::setBVals(const std::vector<T>& bVals)
{
    for (size_t k = 0; k < Layer<T>::out_size; ++k)
    {
        bi(k) = bVals[k];
        bf(k) = bVals[k+Layer<T>::out_size];
        bc(k) = bVals[k+Layer<T>::out_size*2];
        bo(k) = bVals[k+Layer<T>::out_size*3];
    }
}

} // namespace RTNeural
