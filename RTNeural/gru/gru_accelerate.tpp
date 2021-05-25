#include "gru_accelerate.h"

namespace RTNeural
{

template <typename T>
GRULayer<T>::GRULayer(int in_size, int out_size)
    : Layer<T>(in_size, out_size)
    , zWeights(in_size, out_size)
    , rWeights(in_size, out_size)
    , cWeights(in_size, out_size)
{
    ht1 = new T[out_size];
    zVec = new T[out_size];
    rVec = new T[out_size];
    cVec = new T[out_size];
    cTmp = new T[out_size];

    ones = new T[out_size];
    std::fill(ones, &ones[out_size], (T)1);
}

template <typename T>
GRULayer<T>::GRULayer(std::initializer_list<int> sizes)
    : GRULayer<T>(*sizes.begin(), *(sizes.begin() + 1))
{
}

template <typename T>
GRULayer<T>::GRULayer(const GRULayer<T>& other)
    : GRULayer<T>(other.in_size, other.out_size)
{
}

template <typename T>
GRULayer<T>& GRULayer<T>::operator=(const GRULayer<T>& other)
{
    return *this = GRULayer<T>(other);
}

template <typename T>
GRULayer<T>::~GRULayer()
{
    delete[] ht1;
    delete[] zVec;
    delete[] rVec;
    delete[] cVec;
    delete[] cTmp;

    delete[] ones;
}

template <typename T>
GRULayer<T>::WeightSet::WeightSet(int in_size, int out_size)
    : out_size(out_size)
{
    W = new T*[out_size];
    U = new T*[out_size];
    b[0] = new T[out_size];
    b[1] = new T[out_size];

    for(int i = 0; i < out_size; ++i)
    {
        W[i] = new T[in_size];
        U[i] = new T[out_size];
    }
}

template <typename T>
GRULayer<T>::WeightSet::~WeightSet()
{
    delete[] b[0];
    delete[] b[1];

    for(int i = 0; i < out_size; ++i)
    {
        delete[] W[i];
        delete[] U[i];
    }

    delete[] W;
    delete[] U;
}

template <typename T>
void GRULayer<T>::setWVals(const std::vector<std::vector<T>>& wVals)
{
    for(int i = 0; i < Layer<T>::in_size; ++i)
    {
        for(int k = 0; k < Layer<T>::out_size; ++k)
        {
            zWeights.W[k][i] = wVals[i][k];
            rWeights.W[k][i] = wVals[i][k + Layer<T>::out_size];
            cWeights.W[k][i] = wVals[i][k + Layer<T>::out_size * 2];
        }
    }
}

template <typename T>
void GRULayer<T>::setWVals(T** wVals)
{
    for(int i = 0; i < Layer<T>::in_size; ++i)
    {
        for(int k = 0; k < Layer<T>::out_size; ++k)
        {
            zWeights.W[k][i] = wVals[i][k];
            rWeights.W[k][i] = wVals[i][k + Layer<T>::out_size];
            cWeights.W[k][i] = wVals[i][k + Layer<T>::out_size * 2];
        }
    }
}

template <typename T>
void GRULayer<T>::setUVals(const std::vector<std::vector<T>>& uVals)
{
    for(int i = 0; i < Layer<T>::out_size; ++i)
    {
        for(int k = 0; k < Layer<T>::out_size; ++k)
        {
            zWeights.U[k][i] = uVals[i][k];
            rWeights.U[k][i] = uVals[i][k + Layer<T>::out_size];
            cWeights.U[k][i] = uVals[i][k + Layer<T>::out_size * 2];
        }
    }
}

template <typename T>
void GRULayer<T>::setUVals(T** uVals)
{
    for(int i = 0; i < Layer<T>::out_size; ++i)
    {
        for(int k = 0; k < Layer<T>::out_size; ++k)
        {
            zWeights.U[k][i] = uVals[i][k];
            rWeights.U[k][i] = uVals[i][k + Layer<T>::out_size];
            cWeights.U[k][i] = uVals[i][k + Layer<T>::out_size * 2];
        }
    }
}

template <typename T>
void GRULayer<T>::setBVals(const std::vector<std::vector<T>>& bVals)
{
    for(int i = 0; i < 2; ++i)
    {
        for(int k = 0; k < Layer<T>::out_size; ++k)
        {
            zWeights.b[i][k] = bVals[i][k];
            rWeights.b[i][k] = bVals[i][k + Layer<T>::out_size];
            cWeights.b[i][k] = bVals[i][k + Layer<T>::out_size * 2];
        }
    }
}

template <typename T>
void GRULayer<T>::setBVals(T** bVals)
{
    for(int i = 0; i < 2; ++i)
    {
        for(int k = 0; k < Layer<T>::out_size; ++k)
        {
            zWeights.b[i][k] = bVals[i][k];
            rWeights.b[i][k] = bVals[i][k + Layer<T>::out_size];
            cWeights.b[i][k] = bVals[i][k + Layer<T>::out_size * 2];
        }
    }
}

template <typename T>
T GRULayer<T>::getWVal(int i, int k) const noexcept
{
    T** set = zWeights.W;
    if(k > 2 * Layer<T>::out_size)
    {
        k -= 2 * Layer<T>::out_size;
        set = cWeights.W;
    }
    else if(k > Layer<T>::out_size)
    {
        k -= Layer<T>::out_size;
        set = rWeights.W;
    }

    return set[i][k];
}

template <typename T>
T GRULayer<T>::getUVal(int i, int k) const noexcept
{
    T** set = zWeights.U;
    if(k > 2 * Layer<T>::out_size)
    {
        k -= 2 * Layer<T>::out_size;
        set = cWeights.U;
    }
    else if(k > Layer<T>::out_size)
    {
        k -= Layer<T>::out_size;
        set = rWeights.U;
    }

    return set[i][k];
}

template <typename T>
T GRULayer<T>::getBVal(int i, int k) const noexcept
{
    T** set = zWeights.b;
    if(k > 2 * Layer<T>::out_size)
    {
        k -= 2 * Layer<T>::out_size;
        set = cWeights.b;
    }
    else if(k > Layer<T>::out_size)
    {
        k -= Layer<T>::out_size;
        set = rWeights.b;
    }

    return set[i][k];
}

} // namespace RTNeural
