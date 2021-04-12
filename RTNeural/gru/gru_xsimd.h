#ifndef GRUXSIMD_H_INCLUDED
#define GRUXSIMD_H_INCLUDED

#include "../Layer.h"
#include "../common.h"
#include <vector>
namespace RTNeural
{

template <typename T>
class GRULayer : public Layer<T>
{
public:
    GRULayer(size_t in_size, size_t out_size);
    GRULayer(std::initializer_list<size_t> sizes);
    GRULayer(const GRULayer& other);
    GRULayer& operator=(const GRULayer& other);
    virtual ~GRULayer();

    void reset() override { std::fill(ht1.begin(), ht1.end(), (T)0); }

    virtual inline void forward(const T* input, T* h) override
    {
        for(size_t i = 0; i < Layer<T>::out_size; ++i)
        {
            zVec[i] = vMult(zWeights.W[i].data(), input, prod_in.data(), Layer<T>::in_size) + vMult(zWeights.U[i].data(), ht1.data(), prod_out.data(), Layer<T>::out_size);
            rVec[i] = vMult(rWeights.W[i].data(), input, prod_in.data(), Layer<T>::in_size) + vMult(rWeights.U[i].data(), ht1.data(), prod_out.data(), Layer<T>::out_size);
            cVec[i] = vMult(cWeights.W[i].data(), input, prod_in.data(), Layer<T>::in_size);
            cTmp[i] = vMult(cWeights.U[i].data(), ht1.data(), prod_out.data(), Layer<T>::out_size);
        }

        vAdd(zVec.data(), zWeights.b[0].data(), zVec.data(), Layer<T>::out_size);
        vAdd(zVec.data(), zWeights.b[1].data(), zVec.data(), Layer<T>::out_size);
        sigmoid(zVec.data(), zVec.data(), Layer<T>::out_size);

        vAdd(rVec.data(), rWeights.b[0].data(), rVec.data(), Layer<T>::out_size);
        vAdd(rVec.data(), rWeights.b[1].data(), rVec.data(), Layer<T>::out_size);
        sigmoid(rVec.data(), rVec.data(), Layer<T>::out_size);

        vAdd(cTmp.data(), cWeights.b[1].data(), cTmp.data(), Layer<T>::out_size);
        vProd(cTmp.data(), rVec.data(), cTmp.data(), Layer<T>::out_size);
        vAdd(cTmp.data(), cVec.data(), cVec.data(), Layer<T>::out_size);
        vAdd(cVec.data(), cWeights.b[0].data(), cVec.data(), Layer<T>::out_size);
        tanh(cVec.data(), cVec.data(), Layer<T>::out_size);

        vSub(ones.data(), zVec.data(), h, Layer<T>::out_size);
        vProd(h, cVec.data(), h, Layer<T>::out_size);
        vProd(zVec.data(), ht1.data(), prod_out.data(), Layer<T>::out_size);
        vAdd(h, prod_out.data(), h, Layer<T>::out_size);

        vCopy(h, ht1.data(), Layer<T>::out_size);
    }

    void setWVals(T** wVals);
    void setUVals(T** uVals);
    void setBVals(T** bVals);

    void setWVals(const std::vector<std::vector<T>>& wVals);
    void setUVals(const std::vector<std::vector<T>>& uVals);
    void setBVals(const std::vector<std::vector<T>>& bVals);

    T getWVal(size_t i, size_t k) const noexcept;
    T getUVal(size_t i, size_t k) const noexcept;
    T getBVal(size_t i, size_t k) const noexcept;

protected:
    using vec_type = std::vector<T, XSIMD_DEFAULT_ALLOCATOR(T)>;
    using vec2 = std::vector<vec_type>;

    vec_type ht1;

    struct WeightSet
    {
        WeightSet(size_t in_size, size_t out_size);
        ~WeightSet();

        vec2 W;
        vec2 U;
        vec_type b[2];
        const size_t out_size;
    };

    WeightSet zWeights;
    WeightSet rWeights;
    WeightSet cWeights;

    vec_type zVec;
    vec_type rVec;
    vec_type cVec;
    vec_type cTmp;

    vec_type prod_in;
    vec_type prod_out;
    vec_type ones;
};

} // namespace RTNeural

#endif // GRUXSIMD_H_INCLUDED
