#ifndef GRUEIGEN_H_INCLUDED
#define GRUEIGEN_H_INCLUDED

#include "../Layer.h"
#include "../common.h"

namespace RTNeural
{

/**
 * Dynamic implementation of a gated recurrent unit (GRU) layer
 * with tanh activation and sigmoid recurrent activation.
 * 
 * To ensure that the recurrent state is initialized to zero,
 * please make sure to call `reset()` before your first call to
 * the `forward()` method.
 */
template <typename T>
class GRULayer : public Layer<T>
{
public:
    /** Constructs a GRU layer for a given input and output size. */
    GRULayer(int in_size, int out_size);
    GRULayer(std::initializer_list<int> sizes);
    GRULayer(const GRULayer& other);
    GRULayer& operator=(const GRULayer& other);
    virtual ~GRULayer() { }

    /** Resets the state of the GRU. */
    void reset() override
    {
        std::fill(ht1.data(), ht1.data() + Layer<T>::out_size, (T)0);
    }

    /** Returns the name of this layer. */
    std::string getName() const noexcept override { return "gru"; }

    /** Performs forward propagation for this layer. */
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

    /**
     * Sets the layer kernel weights.
     * 
     * The weights vector must have size weights[in_size][3 * out_size]
     */
    void setWVals(T** wVals);

    /**
     * Sets the layer recurrent weights.
     * 
     * The weights vector must have size weights[out_size][3 * out_size]
     */
    void setUVals(T** uVals);

    /**
     * Sets the layer bias.
     * 
     * The bias vector must have size weights[2][3 * out_size]
     */
    void setBVals(T** bVals);

    /** Returns the kernel weight for the given indices. */
    void setWVals(const std::vector<std::vector<T>>& wVals);

    /** Returns the recurrent weight for the given indices. */
    void setUVals(const std::vector<std::vector<T>>& uVals);

    /** Returns the bias value for the given indices. */
    void setBVals(const std::vector<std::vector<T>>& bVals);

    T getWVal(int i, int k) const noexcept;
    T getUVal(int i, int k) const noexcept;
    T getBVal(int i, int k) const noexcept;

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
/**
 * Static implementation of a gated recurrent unit (GRU) layer
 * with tanh activation and sigmoid recurrent activation.
 * 
 * To ensure that the recurrent state is initialized to zero,
 * please make sure to call `reset()` before your first call to
 * the `forward()` method.
 */
template <typename T, int in_sizet, int out_sizet>
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

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "gru"; }

    /** Returns false since GRU is not an activation layer. */
    constexpr bool isActivation() const noexcept { return false; }

    /** Resets the state of the GRU. */
    void reset();

    /** Performs forward propagation for this layer. */
    inline void forward(const in_type& ins)
    {
        zVec = sigmoid(wVec_z * ins + uVec_z * outs + bVec_z);
        rVec = sigmoid(wVec_r * ins + uVec_r * outs + bVec_r);
        cVec = (wVec_c * ins + rVec.cwiseProduct(uVec_c * outs + bVec_c1) + bVec_c0).array().tanh();

        outs = (out_type::Ones() - zVec).cwiseProduct(cVec) + zVec.cwiseProduct(outs);
    }

    /**
     * Sets the layer kernel weights.
     * 
     * The weights vector must have size weights[in_size][3 * out_size]
     */
    void setWVals(const std::vector<std::vector<T>>& wVals);

    /**
     * Sets the layer recurrent weights.
     * 
     * The weights vector must have size weights[out_size][3 * out_size]
     */
    void setUVals(const std::vector<std::vector<T>>& uVals);

    /**
     * Sets the layer bias.
     * 
     * The bias vector must have size weights[2][3 * out_size]
     */
    void setBVals(const std::vector<std::vector<T>>& bVals);

    Eigen::Map<out_type, Eigen::Aligned16> outs;

private:
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];

    static inline out_type sigmoid(const out_type& x) noexcept
    {
        return (T)1 / (((T)-1 * x.array()).array().exp() + (T)1);
    }

    // kernel weights
    k_type wVec_z;
    k_type wVec_r;
    k_type wVec_c;

    // recurrent weights
    r_type uVec_z;
    r_type uVec_r;
    r_type uVec_c;

    // biases
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
