#ifndef LSTM_H_INCLUDED
#define LSTM_H_INCLUDED

#if defined(USE_EIGEN)
// #include "gru_eigen.h"
// #include "gru_eigen.tpp"
#elif defined(USE_XSIMD)
// #include "gru_xsimd.h"
// #include "gru_xsimd.tpp"
#else
#include "common.h"
#include "Layer.h"
#include <vector>

namespace RTNeural
{

template<typename T>
class LSTMLayer : public Layer<T>
{
public:
    LSTMLayer (size_t in_size, size_t out_size);
    virtual ~LSTMLayer();

    void reset() override;

    virtual inline void forward(const T* input, T* h) override
    {
        for(size_t i = 0; i < Layer<T>::out_size; ++i)
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

    struct WeightSet
    {
        WeightSet (size_t in_size, size_t out_size);
        ~WeightSet();

        T** W;
        T** U;
        T* b;
        const size_t out_size;
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

} // namespace RTNeural

#endif

#endif // LSTM_H_INCLUDED
