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

    std::string getName() const noexcept override { return "gru"; }

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
    using vec2_type = std::vector<vec_type>;

    vec_type ht1;

    struct WeightSet
    {
        WeightSet(size_t in_size, size_t out_size);
        ~WeightSet();

        vec2_type W;
        vec2_type U;
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

//====================================================
template<typename T, size_t in_sizet, size_t out_sizet>
class GRULayerT
{
    using v_type = xsimd::simd_type<T>;
    static constexpr auto v_size = v_type::size;
    static constexpr auto v_in_size = ceil_div(in_sizet, v_size);
    static constexpr auto v_out_size = ceil_div(out_sizet, v_size);

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = out_sizet;

    GRULayerT();

    std::string getName() const noexcept { return "gru"; }
    constexpr bool isActivation() const noexcept { return false; }

    void reset();

    template <size_t N = in_size>
    inline typename std::enable_if<(N > 1), void>::type
    forward(const v_type (&ins)[v_in_size])
    {
        // compute zt
        recurrent_mat_mul(outs, Uz, zt);
        kernel_mat_mul(ins, Wz, kernel_outs);
        for(size_t i = 0; i < v_out_size; ++i)
            zt[i] = sigmoid(zt[i] + bz[i] + kernel_outs[i]);

        // compute rt
        recurrent_mat_mul(outs, Ur, rt);
        kernel_mat_mul(ins, Wr, kernel_outs);
        for(size_t i = 0; i < v_out_size; ++i)
            rt[i] = sigmoid(rt[i] + br[i] + kernel_outs[i]);

        // compute h_hat
        recurrent_mat_mul(outs, Uh, ct);
        kernel_mat_mul(ins, Wh, kernel_outs);
        for(size_t i = 0; i < v_out_size; ++i)
            ht[i] = xsimd::tanh(rt[i] * (ct[i] + bh1[i]) + bh0[i] + kernel_outs[i]);

        // compute output
        for(size_t i = 0; i < v_out_size; ++i)
            outs[i] = (v_type ((T) 1.0) - zt[i]) * ht[i] + zt[i] * outs[i];
    }

    template <size_t N = in_size>
    inline typename std::enable_if<N == 1, void>::type
    forward(const v_type (&ins)[v_in_size])
    {
        // compute zt
        recurrent_mat_mul(outs, Uz, zt);
        for(size_t i = 0; i < v_out_size; ++i)
            zt[i] = sigmoid(zt[i] + bz[0] + (Wz_1[i] * ins[0]));

        // compute rt
        recurrent_mat_mul(outs, Ur, rt);
        for(size_t i = 0; i < v_out_size; ++i)
            rt[i] = sigmoid(rt[i] + br[0] + (Wr_1[i] * ins[0]));

        // compute h_hat
        recurrent_mat_mul(outs, Uh, ct);
        for(size_t i = 0; i < v_out_size; ++i)
            ht[i] = xsimd::tanh(rt[i] * (ct[i] + bh1[i]) + bh0[i] + (Wh_1[i] * ins[0]));

        // compute output
        for(size_t i = 0; i < v_out_size; ++i)
            outs[i] = (v_type ((T) 1.0) - zt[i]) * ht[i] + zt[i] * outs[i];
    }

    void setWVals(const std::vector<std::vector<T>>& wVals);
    void setUVals(const std::vector<std::vector<T>>& uVals);
    void setBVals(const std::vector<std::vector<T>>& bVals);

    v_type outs[v_out_size];

private:
    static inline void recurrent_mat_mul(const v_type (&vec)[v_out_size], const v_type (&mat)[out_size][v_out_size], v_type (&out)[v_out_size]) noexcept
    {
        for(size_t i = 0; i < v_size; ++i)
        {
            for(size_t j = 0; j < v_out_size; ++j)
            {
                T sum = (T) 0.0;
                for(size_t k = 0; k < v_out_size; ++k)
                    sum += xsimd::hadd(mat[i + j * v_size][k] * vec[k]);

                out[j] = set_value(out[j], i, sum);
            }
        }
    }

    static inline void kernel_mat_mul(const v_type (&vec)[v_in_size], const v_type (&mat)[out_size][v_in_size], v_type (&out)[v_out_size]) noexcept
    {
        for(size_t i = 0; i < v_size; ++i)
        {
            for(size_t j = 0; j < v_out_size; ++j)
            {
                T sum = (T) 0.0;
                for(size_t k = 0; k < v_in_size; ++k)
                    sum += xsimd::hadd(mat[i + j * v_size][k] * vec[k]);

                out[j] = set_value(out[j], i, sum);
            }
        }
    }

    static inline v_type sigmoid(v_type x) noexcept
    {
        return (T) 1.0 / ((T) 1.0 + xsimd::exp(-x));
    }

    // kernel weights
    v_type Wz[out_size][v_in_size];
    v_type Wr[out_size][v_in_size];
    v_type Wh[out_size][v_in_size];
    v_type kernel_outs[v_out_size];

    // single-input kernel weights
    v_type Wz_1[v_out_size];
    v_type Wr_1[v_out_size];
    v_type Wh_1[v_out_size];

    // recurrent weights
    v_type Uz[out_size][v_out_size];
    v_type Ur[out_size][v_out_size];
    v_type Uh[out_size][v_out_size];

    // biases
    v_type bz[v_out_size];
    v_type br[v_out_size];
    v_type bh0[v_out_size];
    v_type bh1[v_out_size];

    // intermediate vars
    v_type zt[v_out_size];
    v_type rt[v_out_size];
    v_type ct[v_out_size];
    v_type ht[v_out_size];
};

} // namespace RTNeural

#endif // GRUXSIMD_H_INCLUDED
