#ifndef GRUACCELERATE_H_INCLUDED
#define GRUACCELERATE_H_INCLUDED

#include "../Layer.h"
#include "../common.h"
#include <vector>

namespace RTNeural
{

/** Dynamic implementation of a gated recurrent unit (GRU) layer. */
template <typename T>
class GRULayer : public Layer<T>
{
public:
    /** Constructs a GRU layer for a given input and output size. */
    GRULayer(int in_size, int out_size);
    GRULayer(std::initializer_list<int> sizes);
    GRULayer(const GRULayer& other);
    GRULayer& operator=(const GRULayer& other);
    virtual ~GRULayer();

    /** Resets the state of the GRU. */
    void reset() override { std::fill(ht1, ht1 + Layer<T>::out_size, (T)0); }

    /** Returns the name of this layer. */
    std::string getName() const noexcept override { return "gru"; }

    /** Performs forward propagation for this layer. */
    virtual inline void forward(const T* input, T* h) noexcept override
    {
        forward_internal(input, h);
    }

    /** Sets the layer kernel weights. */
    void setWVals(T** wVals);

    /** Sets the layer recurrent weights. */
    void setUVals(T** uVals);

    /** Sets the layer biases. */
    void setBVals(T** bVals);

    /** Sets the layer kernel weights. */
    void setWVals(const std::vector<std::vector<T>>& wVals);

    /** Sets the layer recurrent weights. */
    void setUVals(const std::vector<std::vector<T>>& uVals);

    /** Sets the layer biases. */
    void setBVals(const std::vector<std::vector<T>>& bVals);

    T getWVal(int i, int k) const noexcept;
    T getUVal(int i, int k) const noexcept;
    T getBVal(int i, int k) const noexcept;

protected:
    template <typename FloatType = T>
    inline typename std::enable_if<std::is_same<FloatType, float>::value>::type
    forward_internal(const float* input, float* h) noexcept
    {
        float dotpr_out;
        for(int i = 0; i < Layer<T>::out_size; ++i)
        {
            vDSP_dotpr(zWeights.W[i], 1, input, 1, &dotpr_out, Layer<T>::in_size);
            zVec[i] = dotpr_out;
            vDSP_dotpr(zWeights.U[i], 1, ht1, 1, &dotpr_out, Layer<T>::out_size);
            zVec[i] += dotpr_out;

            vDSP_dotpr(rWeights.W[i], 1, input, 1, &dotpr_out, Layer<T>::in_size);
            rVec[i] = dotpr_out;
            vDSP_dotpr(rWeights.U[i], 1, ht1, 1, &dotpr_out, Layer<T>::out_size);
            rVec[i] += dotpr_out;

            vDSP_dotpr(cWeights.W[i], 1, input, 1, &dotpr_out, Layer<T>::in_size);
            cVec[i] = dotpr_out;
            vDSP_dotpr(cWeights.U[i], 1, ht1, 1, &dotpr_out, Layer<T>::out_size);
            cTmp[i] = dotpr_out;
        }

        vDSP_vadd(zVec, 1, zWeights.b[0], 1, zVec, 1, Layer<T>::out_size);
        vDSP_vadd(zVec, 1, zWeights.b[1], 1, zVec, 1, Layer<T>::out_size);
        sigmoid(zVec, zVec, Layer<T>::out_size);

        vDSP_vadd(rVec, 1, rWeights.b[0], 1, rVec, 1, Layer<T>::out_size);
        vDSP_vadd(rVec, 1, rWeights.b[1], 1, rVec, 1, Layer<T>::out_size);
        sigmoid(rVec, rVec, Layer<T>::out_size);

        vDSP_vadd(cTmp, 1, cWeights.b[1], 1, cTmp, 1, Layer<T>::out_size);
        vDSP_vmul(cTmp, 1, rVec, 1, cTmp, 1, Layer<T>::out_size);
        vDSP_vadd(cTmp, 1, cVec, 1, cVec, 1, Layer<T>::out_size);
        vDSP_vadd(cVec, 1, cWeights.b[0], 1, cVec, 1, Layer<T>::out_size);
        const auto dim_int = static_cast<int>(Layer<T>::out_size);
        vvtanhf(cVec, cVec, &dim_int);

        vDSP_vsub(zVec, 1, ones, 1, h, 1, Layer<T>::out_size);
        vDSP_vmul(h, 1, cVec, 1, h, 1, Layer<T>::out_size);
        vDSP_vmul(zVec, 1, ht1, 1, ht1, 1, Layer<T>::out_size);
        vDSP_vadd(h, 1, ht1, 1, h, 1, Layer<T>::out_size);

        cblas_scopy((int)Layer<T>::out_size, h, 1, ht1, 1);
    }

    template <typename FloatType = T>
    inline typename std::enable_if<std::is_same<FloatType, double>::value>::type
    forward_internal(const double* input, double* h) noexcept
    {
        double dotpr_out;
        for(int i = 0; i < Layer<T>::out_size; ++i)
        {
            vDSP_dotprD(zWeights.W[i], 1, input, 1, &dotpr_out, Layer<T>::in_size);
            zVec[i] = dotpr_out;
            vDSP_dotprD(zWeights.U[i], 1, ht1, 1, &dotpr_out, Layer<T>::out_size);
            zVec[i] += dotpr_out;

            vDSP_dotprD(rWeights.W[i], 1, input, 1, &dotpr_out, Layer<T>::in_size);
            rVec[i] = dotpr_out;
            vDSP_dotprD(rWeights.U[i], 1, ht1, 1, &dotpr_out, Layer<T>::out_size);
            rVec[i] += dotpr_out;

            vDSP_dotprD(cWeights.W[i], 1, input, 1, &dotpr_out, Layer<T>::in_size);
            cVec[i] = dotpr_out;
            vDSP_dotprD(cWeights.U[i], 1, ht1, 1, &dotpr_out, Layer<T>::out_size);
            cTmp[i] = dotpr_out;
        }

        vDSP_vaddD(zVec, 1, zWeights.b[0], 1, zVec, 1, Layer<T>::out_size);
        vDSP_vaddD(zVec, 1, zWeights.b[1], 1, zVec, 1, Layer<T>::out_size);
        sigmoid(zVec, zVec, Layer<T>::out_size);

        vDSP_vaddD(rVec, 1, rWeights.b[0], 1, rVec, 1, Layer<T>::out_size);
        vDSP_vaddD(rVec, 1, rWeights.b[1], 1, rVec, 1, Layer<T>::out_size);
        sigmoid(rVec, rVec, Layer<T>::out_size);

        vDSP_vaddD(cTmp, 1, cWeights.b[1], 1, cTmp, 1, Layer<T>::out_size);
        vDSP_vmulD(cTmp, 1, rVec, 1, cTmp, 1, Layer<T>::out_size);
        vDSP_vaddD(cTmp, 1, cVec, 1, cVec, 1, Layer<T>::out_size);
        vDSP_vaddD(cVec, 1, cWeights.b[0], 1, cVec, 1, Layer<T>::out_size);
        const auto dim_int = static_cast<int>(Layer<T>::out_size);
        vvtanh(cVec, cVec, &dim_int);

        vDSP_vsubD(zVec, 1, ones, 1, h, 1, Layer<T>::out_size);
        vDSP_vmulD(h, 1, cVec, 1, h, 1, Layer<T>::out_size);
        vDSP_vmulD(zVec, 1, ht1, 1, ht1, 1, Layer<T>::out_size);
        vDSP_vaddD(h, 1, ht1, 1, h, 1, Layer<T>::out_size);

        cblas_dcopy((int)Layer<T>::out_size, h, 1, ht1, 1);
    }

    T* ht1;

    struct WeightSet
    {
        WeightSet(int in_size, int out_size);
        ~WeightSet();

        T** W;
        T** U;
        T* b[2];
        const int out_size;
    };

    WeightSet zWeights;
    WeightSet rWeights;
    WeightSet cWeights;

    T* zVec;
    T* rVec;
    T* cVec;
    T* cTmp;

    T* ones;
};

} // namespace RTNeural

#endif // GRUACCELERATE_H_INCLUDED
