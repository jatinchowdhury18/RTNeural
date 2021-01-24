#ifndef GRUXSIMD_H_INCLUDED
#define GRUXSIMD_H_INCLUDED

#include "Layer.h"
#include <vector>
#include <xsimd/xsimd.hpp>

namespace RTNeural
{

template<typename T>
class GRULayer : public Layer<T>
{
public:
    GRULayer (size_t in_size, size_t out_size);
    virtual ~GRULayer();

    virtual void reset()
    {
        std::fill(ht1, ht1 + Layer<T>::out_size, (T) 0);
    }

    virtual inline void forward(const T* input, T* h) override
    {
        for(size_t i = 0; i < Layer<T>::out_size; ++i)
        {
            zVec[i] = vMult(zWeights.W[i], input, prod_in, Layer<T>::in_size) + vMult(zWeights.U[i], ht1, prod_out, Layer<T>::out_size);
            rVec[i] = vMult(rWeights.W[i], input, prod_in, Layer<T>::in_size) + vMult(rWeights.U[i], ht1, prod_out, Layer<T>::out_size);
        }

        vAdd(zVec, zWeights.b[0], zVec, Layer<T>::out_size);
        vAdd(zVec, zWeights.b[1], zVec, Layer<T>::out_size);
        sigmoid(zVec, zVec, Layer<T>::out_size);

        vAdd(rVec, rWeights.b[0], rVec, Layer<T>::out_size);
        vAdd(rVec, rWeights.b[1], rVec, Layer<T>::out_size);
        sigmoid(rVec, rVec, Layer<T>::out_size);

        for(size_t i = 0; i < Layer<T>::out_size; ++i)
            cVec[i] = vMult(cWeights.W[i], input, prod_in, Layer<T>::in_size) + rVec[i] * (vMult(cWeights.U[i], ht1, prod_out, Layer<T>::out_size) + cWeights.b[1][i]);
        vAdd(cVec, cWeights.b[0], cVec, Layer<T>::out_size);
        tanh(cVec, cVec, Layer<T>::out_size);

        vSub(ones, zVec, h, Layer<T>::out_size);
        vProd(h, cVec, h, Layer<T>::out_size);
        vProd(zVec, ht1, prod_out, Layer<T>::out_size);
        vAdd(h, prod_out, h, Layer<T>::out_size);
    
        vCopy(h, ht1, Layer<T>::out_size);
    }

    inline T vMult(const T* arg1, const T* arg2, T* prod, size_t dim) const noexcept
    {
        xsimd::transform(arg1, &arg1[dim], arg2, prod,
            [](auto const &a, auto const &b) { return a * b; });

        return xsimd::reduce (prod, &prod[dim], (T) 0);
    }

    inline void vAdd(const T* in1, const T* in2, T* out, size_t dim) const noexcept
    {
        xsimd::transform(in1, &in1[dim], in2, out,
            [](auto const &a, auto const &b) { return a + b; });
    }

    inline void vSub(const T* in1, const T* in2, T* out, size_t dim) const noexcept
    {
        xsimd::transform(in1, &in1[dim], in2, out,
            [](auto const &a, auto const &b) { return a - b; });
    }

    inline void vProd(const T* in1, const T* in2, T* out, size_t dim) const noexcept
    {
        xsimd::transform(in1, &in1[dim], in2, out,
            [](auto const &a, auto const &b) { return a * b; });
    }

    inline void vCopy(const T* in, T* out, size_t dim) const noexcept
    {
        using b_type = xsimd::simd_type<T>;
        auto inc = b_type::size;

        // size for which the vectorization is possible
        auto vec_size = dim - dim % inc;
        for(size_t i = 0; i < vec_size; i += inc)
        {
            b_type vec = xsimd::load_aligned(&in[i]);
            xsimd::store_aligned(&out[i], vec);
        }
    
        // Remaining part that cannot be vectorize
        for (auto i = vec_size; i < dim; ++i)
            out[i] = in[i];
    }

    inline void sigmoid(const T* in, T* out, size_t dim) const noexcept
    {
        using b_type = xsimd::simd_type<T>;
        auto inc = b_type::size;

        // size for which the vectorization is possible
        auto vec_size = dim - dim % inc;
        for(size_t i = 0; i < vec_size; i += inc)
        {
            b_type x_vec = xsimd::load_aligned(&in[i]);
            b_type y_vec = 1.0 / (1.0 + xsimd::exp(-x_vec));
            xsimd::store_aligned(&out[i], y_vec);
        }
    
        // Remaining part that cannot be vectorize
        for (auto i = vec_size; i < dim; ++i)
            out[i] = 1.0 / (1.0 + std::exp(-in[i]));
    }

    inline void tanh(const T* in, T* out, size_t dim) const noexcept
    {
        using b_type = xsimd::simd_type<T>;
        auto inc = b_type::size;

        // size for which the vectorization is possible
        auto vec_size = dim - dim % inc;
        for(size_t i = 0; i < vec_size; i += inc)
        {
            b_type x_vec = xsimd::load_aligned(&in[i]);
            b_type y_vec = xsimd::tanh(x_vec);
            xsimd::store_aligned(&out[i], y_vec);
        }
    
        // Remaining part that cannot be vectorize
        for (auto i = vec_size; i < dim; ++i)
            out[i] = std::tanh(in[i]);
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
    T* ht1;

    struct WeightSet
    {
        WeightSet (size_t in_size, size_t out_size);
        ~WeightSet();

        T** W;
        T** U;
        T* b[2];
        const size_t out_size;
    };

    WeightSet zWeights;
    WeightSet rWeights;
    WeightSet cWeights;

    T* zVec;
    T* rVec;
    T* cVec;

    T* prod_in;
    T* prod_out;
    T* ones;
};

} // namespace RTNeural

#endif // GRUXSIMD_H_INCLUDED
