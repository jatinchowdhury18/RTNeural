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
    LSTMLayer(std::initializer_list<size_t> sizes);
    LSTMLayer(const LSTMLayer& other);
    LSTMLayer& operator=(const LSTMLayer& other);
    virtual ~LSTMLayer();

    void reset() override;

    std::string getName() const noexcept override { return "lstm"; }

    virtual inline void forward(const T* input, T* h) override
    {
        for(size_t i = 0; i < Layer<T>::out_size; ++i)
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
        WeightSet(size_t in_size, size_t out_size);
        ~WeightSet();

        vec2_type W;
        vec2_type U;
        vec_type b;
        const size_t out_size;
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

} // namespace RTNeural

#endif // LSTM_XSIMD_H_INCLUDED
