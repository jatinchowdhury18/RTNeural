#if RTNEURAL_USE_EIGEN

#include "gru_eigen.h"

namespace RTNeural
{

template <typename T>
GRULayer<T>::GRULayer(int in_size, int out_size)
    : Layer<T>(in_size, out_size)
{
    /*
     * | Wz Rz bz0 bz1 |
     * | Wr Rr br0 br1 |
     */
    combinedWeights_rz = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(2 * out_size, in_size + out_size + 2);

    /*
     * | Wc bc0 |
     */
    combinedWeights_cx = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, in_size + 1);

    /*
     * | Rc bc1 |
     */
    combinedWeights_ch = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, out_size + 1);

    zrVec = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(2 * out_size, 1);

    ht1_temp = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(out_size, 1);
    ht1      = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(out_size + 1, 1);
    ht1(out_size, 0) = (T)1;

    cVec = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(out_size, 1);

    inVec = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(in_size + 1, 1);
    inVec(in_size, 0) = (T)1;

    /*
     * | xt h(t-1) 1 1 |
     */
    xVec = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(in_size + out_size + 2, 1);
    xVec.tail(2) = (T)1;
}

template <typename T>
GRULayer<T>::GRULayer(std::initializer_list<int> sizes)
    : GRULayer<T>(*sizes.begin(), *(sizes.begin() + 1))
{
}

template <typename T>
GRULayer<T>::GRULayer(const GRULayer<T>& other)
    : GRULayer<T>(other.in_size, other.out_size)
{
}

template <typename T>
GRULayer<T>& GRULayer<T>::operator=(const GRULayer<T>& other)
{
    return *this = GRULayer<T>(other);
}

template <typename T>
void GRULayer<T>::setWVals(const std::vector<std::vector<T>>& wVals)
{
    for(int i = 0; i < Layer<T>::in_size; ++i)
    {
        for(int k = 0; k < Layer<T>::out_size; ++k)
        {
            combinedWeights_rz(k, i) = wVals[i][k];
            combinedWeights_rz(k + Layer<T>::out_size, i) = wVals[i][k + Layer<T>::out_size];
            combinedWeights_cx(k, i) = wVals[i][k + Layer<T>::out_size * 2];
        }
    }
}

template <typename T>
void GRULayer<T>::setWVals(T** wVals)
{
    for(int i = 0; i < Layer<T>::in_size; ++i)
    {
        for(int k = 0; k < Layer<T>::out_size; ++k)
        {
            combinedWeights_rz(k, i) = wVals[i][k];
            combinedWeights_rz(k + Layer<T>::out_size, i) = wVals[i][k + Layer<T>::out_size];
            combinedWeights_cx(k, i) = wVals[i][k + Layer<T>::out_size * 2];
        }
    }
}

template <typename T>
void GRULayer<T>::setUVals(const std::vector<std::vector<T>>& uVals)
{
    int col;
    for(int i = 0; i < Layer<T>::out_size; ++i)
    {
        col = i + Layer<T>::in_size;
        for(int k = 0; k < Layer<T>::out_size; ++k)
        {
            combinedWeights_rz(k,                      col) = uVals[i][k];
            combinedWeights_rz(k + Layer<T>::out_size, col) = uVals[i][k + Layer<T>::out_size];
            combinedWeights_ch(k, i) = uVals[i][k + Layer<T>::out_size * 2];
        }
    }
}

template <typename T>
void GRULayer<T>::setUVals(T** uVals)
{
    int col;
    for(int i = 0; i < Layer<T>::out_size; ++i)
    {
        col = i + Layer<T>::in_size;
        for(int k = 0; k < Layer<T>::out_size; ++k)
        {
            combinedWeights_rz(k,                      col) = uVals[i][k];
            combinedWeights_rz(k + Layer<T>::out_size, col) = uVals[i][k + Layer<T>::out_size];
            combinedWeights_ch(k, i) = uVals[i][k + Layer<T>::out_size * 2];
        }
    }
}

template <typename T>
void GRULayer<T>::setBVals(const std::vector<std::vector<T>>& bVals)
{
    int col;
    for(int i = 0; i < 2; ++i)
    {
        col = i + Layer<T>::in_size + Layer<T>::out_size;
        for(int k = 0; k < Layer<T>::out_size; ++k)
        {
            combinedWeights_rz(k,                      col) = bVals[i][k];
            combinedWeights_rz(k + Layer<T>::out_size, col) = bVals[i][k + Layer<T>::out_size];
            if (i == 0)
                combinedWeights_cx(k, Layer<T>::in_size) = bVals[i][k + Layer<T>::out_size * 2];
            else
                combinedWeights_ch(k, Layer<T>::out_size) = bVals[i][k + Layer<T>::out_size * 2];
        }
    }
}

template <typename T>
void GRULayer<T>::setBVals(T** bVals)
{
    int col;
    for(int i = 0; i < 2; ++i)
    {
        col = i + Layer<T>::in_size + Layer<T>::out_size;
        for(int k = 0; k < Layer<T>::out_size; ++k)
        {
            combinedWeights_rz(k,                      col) = bVals[i][k];
            combinedWeights_rz(k + Layer<T>::out_size, col) = bVals[i][k + Layer<T>::out_size];
            if (i == 0)
                combinedWeights_cx(k, i + Layer<T>::in_size) = bVals[i][k + Layer<T>::out_size * 2];
            else
                combinedWeights_ch(k, i + Layer<T>::out_size) = bVals[i][k + Layer<T>::out_size * 2];
        }
    }
}

template <typename T>
T GRULayer<T>::getWVal(int i, int k) const noexcept
{
    T val;
    if (k < 2 * Layer<T>::out_size)
        val = combinedWeights_rz(k, i);
    else
        val = combinedWeights_cx(k % Layer<T>::out_size, i);
    return val;
}

template <typename T>
T GRULayer<T>::getUVal(int i, int k) const noexcept
{
    T val;
    if (k < 2 * Layer<T>::out_size)
        val = combinedWeights_rz(k, i + Layer<T>::in_size);
    else
        val = combinedWeights_ch(k % Layer<T>::out_size, i + Layer<T>::in_size);
    return val;
}

template <typename T>
T GRULayer<T>::getBVal(int i, int k) const noexcept
{
    T val;
    if (k < 2 * Layer<T>::out_size)
    {
        val = combinedWeights_rz(k, i + Layer<T>::in_size + Layer<T>::out_size);
    }
    else
    {
        if (i == 0)
            val = combinedWeights_cx(k % Layer<T>::out_size, Layer<T>::in_size);
        else
            val = combinedWeights_ch(k % Layer<T>::out_size, Layer<T>::out_size);
    }
    return val;
}

//====================================================
template <typename T, int in_sizet, int out_sizet, SampleRateCorrectionMode sampleRateCorr>
GRULayerT<T, in_sizet, out_sizet, sampleRateCorr>::GRULayerT()
    : outs(outs_internal)
{
    wVec_zr = zr_mat_type::Zero();
    xVec = xh_vec_type::Zero();
    xVec.tail(2) = (T)1;
    zrVec = zr_vec_type::Zero();

    wVec_cx = cx_mat_type::Zero();
    wVec_ch = ch_mat_type::Zero();
    cVec = out_type::Zero();

    reset();
}

template <typename T, int in_sizet, int out_sizet, SampleRateCorrectionMode sampleRateCorr>
template <SampleRateCorrectionMode srCorr>
std::enable_if_t<srCorr == SampleRateCorrectionMode::NoInterp, void>
GRULayerT<T, in_sizet, out_sizet, sampleRateCorr>::prepare(int delaySamples)
{
    delayWriteIdx = delaySamples - 1;
    outs_delayed.resize(delayWriteIdx + 1, {});

    reset();
}

template <typename T, int in_sizet, int out_sizet, SampleRateCorrectionMode sampleRateCorr>
template <SampleRateCorrectionMode srCorr>
std::enable_if_t<srCorr == SampleRateCorrectionMode::LinInterp, void>
GRULayerT<T, in_sizet, out_sizet, sampleRateCorr>::prepare(T delaySamples)
{
    const auto delayOffFactor = delaySamples - std::floor(delaySamples);
    delayMult = (T)1 - delayOffFactor;
    delayPlus1Mult = delayOffFactor;

    delayWriteIdx = (int)std::ceil(delaySamples) - (int)std::ceil(delayOffFactor);
    outs_delayed.resize(delayWriteIdx + 1, {});

    reset();
}

template <typename T, int in_sizet, int out_sizet, SampleRateCorrectionMode sampleRateCorr>
void GRULayerT<T, in_sizet, out_sizet, sampleRateCorr>::reset()
{
    if(sampleRateCorr != SampleRateCorrectionMode::None)
    {
        for(auto& vec : outs_delayed)
            vec = out_type::Zero();
    }

    // reset output state
    outs = out_type::Zero();
}

// kernel weights
template <typename T, int in_sizet, int out_sizet, SampleRateCorrectionMode sampleRateCorr>
void GRULayerT<T, in_sizet, out_sizet, sampleRateCorr>::setWVals(const std::vector<std::vector<T>>& wVals)
{
    for(int i = 0; i < in_size; ++i)
    {
        for(int k = 0; k < out_size; ++k)
        {
            wVec_z(k, i) = wVals[i][k];
            wVec_r(k, i) = wVals[i][k + out_size];
            wVec_c(k, i) = wVals[i][k + out_size * 2];
        }
    }
}

// recurrent weights
template <typename T, int in_sizet, int out_sizet, SampleRateCorrectionMode sampleRateCorr>
void GRULayerT<T, in_sizet, out_sizet, sampleRateCorr>::setUVals(const std::vector<std::vector<T>>& uVals)
{
    for(int i = 0; i < out_size; ++i)
    {
        for(int k = 0; k < out_size; ++k)
        {
            uVec_z(k, i) = uVals[i][k];
            uVec_r(k, i) = uVals[i][k + out_size];
            uVec_c(k, i) = uVals[i][k + out_size * 2];
        }
    }
}

// biases
template <typename T, int in_sizet, int out_sizet, SampleRateCorrectionMode sampleRateCorr>
void GRULayerT<T, in_sizet, out_sizet, sampleRateCorr>::setBVals(const std::vector<std::vector<T>>& bVals)
{
    for(int k = 0; k < out_size; ++k)
    {
        bVec_z(k) = bVals[0][k] + bVals[1][k];
        bVec_r(k) = bVals[0][k + out_size] + bVals[1][k + out_size];
        bVec_c0(k) = bVals[0][k + 2 * out_size];
        bVec_c1(k) = bVals[1][k + 2 * out_size];
    }
}

} // namespace RTNeural

#endif // RTNEURAL_USE_EIGEN
