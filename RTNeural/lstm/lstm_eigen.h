#ifndef LSTM_EIGEN_INCLUDED
#define LSTM_EIGEN_INCLUDED

#include "../Layer.h"
#include "../common.h"

namespace RTNeural
{

/**
 * Dynamic implementation of a LSTM layer with tanh
 * activation and sigmoid recurrent activation.
 *
 * To ensure that the recurrent state is initialized to zero,
 * please make sure to call `reset()` before your first call to
 * the `forward()` method.
 */
template <typename T>
class LSTMLayer : public Layer<T>
{
public:
    /** Constructs a LSTM layer for a given input and output size. */
    LSTMLayer(int in_size, int out_size);
    LSTMLayer(std::initializer_list<int> sizes);
    LSTMLayer(const LSTMLayer& other);
    LSTMLayer& operator=(const LSTMLayer& other);
    virtual ~LSTMLayer() = default;

    /** Returns the name of this layer. */
    std::string getName() const noexcept override { return "lstm"; }

    /** Resets the state of the LSTM. */
    void reset() override;

    /** Performs forward propagation for this layer. */
    inline void forward(const T* input, T* h) noexcept override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, RTNeuralEigenAlignment>(
            input, Layer<T>::in_size, 1);

        fVec.noalias() = Wf * inVec + Uf * ht1 + bf;
        iVec.noalias() = Wi * inVec + Ui * ht1 + bi;
        oVec.noalias() = Wo * inVec + Uo * ht1 + bo;

        ctVec.noalias() = Wc * inVec + Uc * ht1 + bc;
        ctVec = ctVec.array().tanh();

        sigmoid(fVec);
        sigmoid(iVec);
        sigmoid(oVec);

        cVec.noalias() = fVec.cwiseProduct(ct1) + iVec.cwiseProduct(ctVec);
        ht1 = cVec.array().tanh();
        ht1 = oVec.cwiseProduct(ht1);

        ct1 = cVec;
        std::copy(ht1.data(), ht1.data() + Layer<T>::out_size, h);
    }

    /**
     * Sets the layer kernel weights.
     *
     * The weights vector must have size weights[in_size][4 * out_size]
     */
    void setWVals(const std::vector<std::vector<T>>& wVals);

    /**
     * Sets the layer recurrent weights.
     *
     * The weights vector must have size weights[out_size][4 * out_size]
     */
    void setUVals(const std::vector<std::vector<T>>& uVals);

    /**
     * Sets the layer bias.
     *
     * The bias vector must have size weights[4 * out_size]
     */
    void setBVals(const std::vector<T>& bVals);

private:
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Wf;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Wi;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Wo;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Wc;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Uf;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Ui;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Uo;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Uc;
    Eigen::Matrix<T, Eigen::Dynamic, 1> bf;
    Eigen::Matrix<T, Eigen::Dynamic, 1> bi;
    Eigen::Matrix<T, Eigen::Dynamic, 1> bo;
    Eigen::Matrix<T, Eigen::Dynamic, 1> bc;

    Eigen::Matrix<T, Eigen::Dynamic, 1> fVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> iVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> oVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> ctVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> cVec;

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> ht1;
    Eigen::Matrix<T, Eigen::Dynamic, 1> ct1;
};

//====================================================
/**
 * Static implementation of a LSTM layer with tanh
 * activation and sigmoid recurrent activation.
 *
 * To ensure that the recurrent state is initialized to zero,
 * please make sure to call `reset()` before your first call to
 * the `forward()` method.
 */
template <typename T, int in_sizet, int out_sizet, SampleRateCorrectionMode sampleRateCorr = SampleRateCorrectionMode::None>
class LSTMLayerT
{
    using b_type = Eigen::Matrix<T, out_sizet, 1>;
    using k_type = Eigen::Matrix<T, out_sizet, in_sizet>;
    using r_type = Eigen::Matrix<T, out_sizet, out_sizet>;

    using in_type = Eigen::Matrix<T, in_sizet, 1>;
    using out_type = Eigen::Matrix<T, out_sizet, 1>;

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = out_sizet;

    LSTMLayerT();

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "lstm"; }

    /** Returns false since LSTM is not an activation. */
    constexpr bool isActivation() const noexcept { return false; }

    /** Prepares the LSTM to process with a given delay length. */
    template <SampleRateCorrectionMode srCorr = sampleRateCorr>
    std::enable_if_t<srCorr == SampleRateCorrectionMode::NoInterp, void>
    prepare(int delaySamples);

    /** Prepares the LSTM to process with a given delay length. */
    template <SampleRateCorrectionMode srCorr = sampleRateCorr>
    std::enable_if_t<srCorr == SampleRateCorrectionMode::LinInterp, void>
    prepare(T delaySamples);

    /** Resets the state of the LSTM. */
    void reset();

    /** Performs forward propagation for this layer. */
    inline void forward(const in_type& ins) noexcept
    {
        fVec.noalias() = bf;
        fVec.noalias() += Uf * outs;
        fVec.noalias() += Wf * ins;

        iVec.noalias() = bi;
        iVec.noalias() += Ui * outs;
        iVec.noalias() += Wi * ins;

        oVec.noalias() = bo;
        oVec.noalias() += Uo * outs;
        oVec.noalias() += Wo * ins;

        fVec = sigmoid(fVec);
        iVec = sigmoid(iVec);
        oVec = sigmoid(oVec);

        computeOutputs(ins);
    }

    /**
     * Sets the layer kernel weights.
     *
     * The weights vector must have size weights[in_size][4 * out_size]
     */
    void setWVals(const std::vector<std::vector<T>>& wVals);

    /**
     * Sets the layer recurrent weights.
     *
     * The weights vector must have size weights[out_size][4 * out_size]
     */
    void setUVals(const std::vector<std::vector<T>>& uVals);

    /**
     * Sets the layer bias.
     *
     * The bias vector must have size weights[4 * out_size]
     */
    void setBVals(const std::vector<T>& bVals);

    Eigen::Map<out_type, RTNeuralEigenAlignment> outs;

private:
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];

    template <SampleRateCorrectionMode srCorr = sampleRateCorr>
    inline std::enable_if_t<srCorr == SampleRateCorrectionMode::None, void>
    computeOutputs(const in_type& ins) noexcept
    {
        computeOutputsInternal(ins, cVec, outs);
    }

    template <SampleRateCorrectionMode srCorr = sampleRateCorr>
    inline std::enable_if_t<srCorr != SampleRateCorrectionMode::None, void>
    computeOutputs(const in_type& ins) noexcept
    {
        computeOutputsInternal(ins, ct_delayed[delayWriteIdx], outs_delayed[delayWriteIdx]);

        processDelay(ct_delayed, cVec, delayWriteIdx);
        processDelay(outs_delayed, outs, delayWriteIdx);
    }

    template <typename VecType1, typename VecType2>
    inline void computeOutputsInternal(const in_type& ins, VecType1& cVecLocal, VecType2& outsVec) noexcept
    {
        ctVec.noalias() = bc;
        ctVec.noalias() += Uc * outs;
        ctVec.noalias() += Wc * ins;

        ctVec = ctVec.array().tanh();
        cVecLocal = fVec.cwiseProduct(cVec);
        cVecLocal.noalias() += iVec.cwiseProduct(ctVec);

        outsVec = cVecLocal.array().tanh();
        outsVec = oVec.cwiseProduct(outsVec);
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
    k_type Wf;
    k_type Wi;
    k_type Wo;
    k_type Wc;

    // recurrent weights
    r_type Uf;
    r_type Ui;
    r_type Uo;
    r_type Uc;

    // biases
    b_type bf;
    b_type bi;
    b_type bo;
    b_type bc;

    // intermediate values
    out_type fVec;
    out_type iVec;
    out_type oVec;
    out_type ctVec;
    out_type cVec;

    // needed for delays when doing sample rate correction
    std::vector<out_type> ct_delayed;
    std::vector<out_type> outs_delayed;
    int delayWriteIdx = 0;
    T delayMult = (T)1;
    T delayPlus1Mult = (T)0;
};

} // namespace RTNeural

#endif // LSTM_EIGEN_INCLUDED
