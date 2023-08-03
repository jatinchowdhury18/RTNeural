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
    virtual ~GRULayer() = default;

    /** Resets the state of the GRU. */
    void reset() override
    {
        std::fill(ht1.data(), ht1.data() + Layer<T>::out_size, (T)0);
        std::fill(ht1_temp.data(), ht1_temp.data() + Layer<T>::out_size, (T)0);
        std::fill(xVec.data() + Layer<T>::in_size, xVec.data() + Layer<T>::in_size + Layer<T>::out_size, (T)0);
    }

    /** Returns the name of this layer. */
    std::string getName() const noexcept override { return "gru"; }

    /** Performs forward propagation for this layer. */
    inline void forward(const T* input, T* h) noexcept override
    {
//        memcpy(inVec.data(), input, Layer<T>::in_size);
//        memcpy(xVec.data(),  input, Layer<T>::in_size);

        // No need to copy h(t-1) because it's already in xVec from the previous iteration or from instantiation/reset
        for (int i = 0; i < Layer<T>::in_size; ++i)
        {
            xVec(i) = inVec(i) = input[i];
        }

        zrVec.noalias() = combinedWeights_rz * xVec;
        sigmoid(zrVec);

        cVec.noalias() = combinedWeights_cx * inVec +
                         zrVec.segment(Layer<T>::out_size, Layer<T>::out_size).cwiseProduct(combinedWeights_ch * ht1);
        cVec = cVec.array().tanh();

        ht1_temp = cVec + zrVec.segment(0, Layer<T>::out_size).cwiseProduct(ht1_temp - cVec);

        for (int i = 0; i < Layer<T>::out_size; ++i)
        {
            h[i] = xVec(i + Layer<T>::in_size) = ht1(i) = ht1_temp(i);
        }
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

    Eigen::Matrix<T, Eigen::Dynamic, 1> ht1;
    Eigen::Matrix<T, Eigen::Dynamic, 1> ht1_temp;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> combinedWeights_rz;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> combinedWeights_cx;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> combinedWeights_ch;

    Eigen::Matrix<T, Eigen::Dynamic, 1> zrVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> cVec;

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> xVec;
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
template <typename T, int in_sizet, int out_sizet, SampleRateCorrectionMode sampleRateCorr = SampleRateCorrectionMode::None>
class GRULayerT
{
    using b_type = Eigen::Matrix<T, out_sizet, 1>;

    using zr_mat_type = Eigen::Matrix<T, 2 * out_sizet, in_sizet + out_sizet + 2>;
    using xh_vec_type = Eigen::Matrix<T, in_sizet + out_sizet + 2, 1>;
    using zr_vec_type = Eigen::Matrix<T, 2 * out_sizet, 1>;

    using cx_mat_type = Eigen::Matrix<T, out_sizet, in_sizet + 1>;
    using ch_mat_type = Eigen::Matrix<T, out_sizet, out_sizet + 1>;

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

    /** Prepares the GRU to process with a given delay length. */
    template <SampleRateCorrectionMode srCorr = sampleRateCorr>
    std::enable_if_t<srCorr == SampleRateCorrectionMode::NoInterp, void>
    prepare(int delaySamples);

    /** Prepares the GRU to process with a given delay length. */
    template <SampleRateCorrectionMode srCorr = sampleRateCorr>
    std::enable_if_t<srCorr == SampleRateCorrectionMode::LinInterp, void>
    prepare(T delaySamples);

    /** Resets the state of the GRU. */
    void reset();

    /** Performs forward propagation for this layer. */
    inline void forward(const in_type& ins) noexcept
    {
        zVec.noalias() = sigmoid(wVec_z * ins + uVec_z * outs + bVec_z);
        rVec.noalias() = sigmoid(wVec_r * ins + uVec_r * outs + bVec_r);

        cVec.noalias() = wVec_c * ins + bVec_c0 + rVec.cwiseProduct(uVec_c * outs + bVec_c1);
        cVec = cVec.array().tanh();

        computeOutput();
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

    Eigen::Map<out_type, RTNeuralEigenAlignment> outs;

private:
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];

    template <SampleRateCorrectionMode srCorr = sampleRateCorr>
    inline std::enable_if_t<srCorr == SampleRateCorrectionMode::None, void>
    computeOutput() noexcept
    {
        outs = cVec + zrVec.segment(0, out_sizet).cwiseProduct(outs - cVec);
    }

    template <SampleRateCorrectionMode srCorr = sampleRateCorr>
    inline std::enable_if_t<srCorr != SampleRateCorrectionMode::None, void>
    computeOutput() noexcept
    {
        outs_delayed[delayWriteIdx] = cVec + zrVec.segment(0, out_sizet).cwiseProduct(outs - cVec);

        processDelay(outs_delayed, outs, delayWriteIdx);
    }

    template <typename OutVec, SampleRateCorrectionMode srCorr = sampleRateCorr>
    inline std::enable_if_t<srCorr == SampleRateCorrectionMode::NoInterp, void>
    processDelay(std::vector<out_type>& delayVec, OutVec& out, int delayWriteIndex) noexcept
    {
        out = delayVec[0];

        for(int j = 0; j < delayWriteIndex; ++j)
            delayVec[j] = delayVec[j + 1];
    }

    template <typename OutVec, SampleRateCorrectionMode srCorr = sampleRateCorr>
    inline std::enable_if_t<srCorr == SampleRateCorrectionMode::LinInterp, void>
    processDelay(std::vector<out_type>& delayVec, OutVec& out, int delayWriteIndex) noexcept
    {
        out = delayPlus1Mult * delayVec[0] + delayMult * delayVec[1];

        for(int j = 0; j < delayWriteIndex; ++j)
            delayVec[j] = delayVec[j + 1];
    }

    static inline out_type sigmoid(const out_type& x) noexcept
    {
        return (T)1 / (((T)-1 * x.array()).array().exp() + (T)1);
    }

    // kernel weights
    zr_mat_type wVec_zr;
    cx_mat_type wVec_cx;
    ch_mat_type wVec_ch;

    // scratch memory
    zr_vec_type zrVec;
    xh_vec_type xVec;
    out_type cVec;

    // needed for delays when doing sample rate correction
    std::vector<out_type> outs_delayed;
    int delayWriteIdx = 0;
    T delayMult = (T)1;
    T delayPlus1Mult = (T)0;
};

} // namespace RTNeural

#endif // GRUEIGEN_H_INCLUDED
