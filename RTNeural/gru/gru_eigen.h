#ifndef GRUEIGEN_H_INCLUDED
#define GRUEIGEN_H_INCLUDED

#include "../Layer.h"
#include "../common.h"

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
    virtual ~GRULayer() { }

    void reset() override
    {
        std::fill(ht1.data(), ht1.data() + Layer<T>::out_size, (T)0);
    }

    std::string getName() const noexcept override { return "gru"; }

    inline void forward(const T* input, T* h) override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::Aligned16>(
            input, Layer<T>::in_size, 1);

        zVec = wVec_z * inVec + uVec_z * ht1 + bVec_z.col(0) + bVec_z.col(1);
        rVec = wVec_r * inVec + uVec_r * ht1 + bVec_r.col(0) + bVec_r.col(1);
        sigmoid(zVec);
        sigmoid(rVec);

        cVec = wVec_c * inVec + rVec.cwiseProduct(uVec_c * ht1 + bVec_c.col(1)) + bVec_c.col(0);
        cVec = cVec.array().tanh();

        ht1 = (ones - zVec).cwiseProduct(cVec) + zVec.cwiseProduct(ht1);
        std::copy(ht1.data(), ht1.data() + Layer<T>::out_size, h);
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

private:
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> wVec_z;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> wVec_r;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> wVec_c;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> uVec_z;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> uVec_r;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> uVec_c;
    Eigen::Matrix<T, Eigen::Dynamic, 2> bVec_z;
    Eigen::Matrix<T, Eigen::Dynamic, 2> bVec_r;
    Eigen::Matrix<T, Eigen::Dynamic, 2> bVec_c;

    Eigen::Matrix<T, Eigen::Dynamic, 1> ht1;
    Eigen::Matrix<T, Eigen::Dynamic, 1> zVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> rVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> cVec;

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> ones;
};

//====================================================
template <typename T, size_t in_sizet, size_t out_sizet>
class GRULayerT
{
    using b_type = Eigen::Matrix<T, out_sizet, 1>;
    using k_type = Eigen::Matrix<T, out_sizet, in_sizet>;
    using r_type = Eigen::Matrix<T, out_sizet, out_sizet>;

    using in_type = Eigen::Matrix<T, in_sizet, 1>;
    using out_type = Eigen::Matrix<T, out_sizet, 1>;

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = out_sizet;

    GRULayerT();

    std::string getName() const noexcept { return "gru"; }
    constexpr bool isActivation() const noexcept { return false; }

    void reset();

    inline void forward(const in_type& ins)
    {
        zVec = sigmoid(wVec_z * ins + uVec_z * outs + bVec_z);
        rVec = sigmoid(wVec_r * ins + uVec_r * outs + bVec_r);
        cVec = (wVec_c * ins + rVec.cwiseProduct(uVec_c * outs + bVec_c1) + bVec_c0).array().tanh();

        outs = (out_type::Ones() - zVec).cwiseProduct(cVec) + zVec.cwiseProduct(outs);
    }

    void setWVals(const std::vector<std::vector<T>>& wVals);
    void setUVals(const std::vector<std::vector<T>>& uVals);
    void setBVals(const std::vector<std::vector<T>>& bVals);

    Eigen::Map<out_type, Eigen::Aligned16> outs;

private:
    T outs_internal alignas(16)[out_size];

    static inline out_type sigmoid(const out_type& x) noexcept
    {
        return (T)1 / (((T)-1 * x.array()).array().exp() + (T)1);
    }

    k_type wVec_z;
    k_type wVec_r;
    k_type wVec_c;

    r_type uVec_z;
    r_type uVec_r;
    r_type uVec_c;

    b_type bVec_z;
    b_type bVec_r;
    b_type bVec_c0;
    b_type bVec_c1;

    out_type zVec;
    out_type rVec;
    out_type cVec;
};

} // namespace RTNeural

#endif // GRUEIGEN_H_INCLUDED
