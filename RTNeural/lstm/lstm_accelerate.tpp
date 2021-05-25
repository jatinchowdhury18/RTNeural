#include "lstm_accelerate.h"

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
    ht1 = new T[out_size];
    ct1 = new T[out_size];

    fVec = new T[out_size];
    iVec = new T[out_size];
    oVec = new T[out_size];
    ctVec = new T[out_size];
    cVec = new T[out_size];
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
    delete[] ht1;
    delete[] ct1;

    delete[] fVec;
    delete[] iVec;
    delete[] oVec;
    delete[] ctVec;
    delete[] cVec;
}

template <typename T>
void LSTMLayer<T>::reset()
{
    std::fill(ht1, ht1 + Layer<T>::out_size, (T)0);
    std::fill(ct1, ct1 + Layer<T>::out_size, (T)0);
}

template <typename T>
LSTMLayer<T>::WeightSet::WeightSet(int in_size, int out_size)
    : out_size(out_size)
{
    W = new T*[out_size];
    U = new T*[out_size];
    b = new T[out_size];

    for(int i = 0; i < out_size; ++i)
    {
        W[i] = new T[in_size];
        U[i] = new T[out_size];
    }
}

template <typename T>
LSTMLayer<T>::WeightSet::~WeightSet()
{
    delete[] b;

    for(int i = 0; i < out_size; ++i)
    {
        delete[] W[i];
        delete[] U[i];
    }

    delete[] W;
    delete[] U;
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

} // namespace RTNeural
