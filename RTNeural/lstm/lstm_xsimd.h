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
    LSTMLayer(size_t in_size, size_t out_size);
    virtual ~LSTMLayer();

    void reset() override;
    virtual inline void forward(const T* input, T* h) override
    {
        for(size_t i = 0; i < Layer<T>::out_size; ++i)
        {
            fVec[i] = vMult(fWeights.W[i], input, prod_in, Layer<T>::in_size) + vMult(fWeights.U[i], ht1, prod_out, Layer<T>::out_size);
            iVec[i] = vMult(iWeights.W[i], input, prod_in, Layer<T>::in_size) + vMult(iWeights.U[i], ht1, prod_out, Layer<T>::out_size);
            oVec[i] = vMult(oWeights.W[i], input, prod_in, Layer<T>::in_size) + vMult(oWeights.U[i], ht1, prod_out, Layer<T>::out_size);
            ctVec[i] = vMult(cWeights.W[i], input, prod_in, Layer<T>::in_size) + vMult(cWeights.U[i], ht1, prod_out, Layer<T>::out_size);
        }

        vAdd(fVec, fWeights.b, fVec, Layer<T>::out_size);
        sigmoid(fVec, fVec, Layer<T>::out_size);

        vAdd(iVec, iWeights.b, iVec, Layer<T>::out_size);
        sigmoid(iVec, iVec, Layer<T>::out_size);

        vAdd(oVec, oWeights.b, oVec, Layer<T>::out_size);
        sigmoid(oVec, oVec, Layer<T>::out_size);

        vAdd(ctVec, cWeights.b, ctVec, Layer<T>::out_size);
        tanh(ctVec, ctVec, Layer<T>::out_size);

        vProd(fVec, ct1, cVec, Layer<T>::out_size);
        vProd(iVec, ctVec, prod_out, Layer<T>::out_size);
        vAdd(cVec, prod_out, cVec, Layer<T>::out_size);

        tanh(cVec, h, Layer<T>::out_size);
        vMult(h, oVec, h, Layer<T>::out_size);

        vCopy(cVec, ct1, Layer<T>::out_size);
        vCopy(h, ht1, Layer<T>::out_size);
    }

    void setWVals(const std::vector<std::vector<T>>& wVals);
    void setUVals(const std::vector<std::vector<T>>& uVals);
    void setBVals(const std::vector<T>& bVals);

protected:
    T* ht1;
    T* ct1;

    struct WeightSet
    {
        WeightSet(size_t in_size, size_t out_size);
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

    T* prod_in;
    T* prod_out;
};

} // namespace RTNeural

#endif // LSTM_XSIMD_H_INCLUDED
