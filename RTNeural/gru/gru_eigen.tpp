#if RTNEURAL_USE_EIGEN

#include "gru_eigen.h"

namespace RTNeural
{

template <typename T>
GRULayer<T>::GRULayer(int in_size, int out_size)
    : Layer<T>(in_size, out_size)
{
    wVec_z = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, in_size);
    wVec_r = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, in_size);
    wVec_c = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, in_size);
    uVec_z = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, out_size);
    uVec_r = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, out_size);
    uVec_c = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, out_size);
    bVec_z = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, 2);
    bVec_r = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, 2);
    bVec_c = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, 2);

    ht1 = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, 1);
    zVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, 1);
    rVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, 1);
    cVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, 1);

    inVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(in_size, 1);
    ones = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Ones(out_size, 1);
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
            wVec_z(k, i) = wVals[i][k];
            wVec_r(k, i) = wVals[i][k + Layer<T>::out_size];
            wVec_c(k, i) = wVals[i][k + Layer<T>::out_size * 2];
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
            wVec_z(k, i) = wVals[i][k];
            wVec_r(k, i) = wVals[i][k + Layer<T>::out_size];
            wVec_c(k, i) = wVals[i][k + Layer<T>::out_size * 2];
        }
    }
}

template <typename T>
void GRULayer<T>::setUVals(const std::vector<std::vector<T>>& uVals)
{
    for(int i = 0; i < Layer<T>::out_size; ++i)
    {
        for(int k = 0; k < Layer<T>::out_size; ++k)
        {
            uVec_z(k, i) = uVals[i][k];
            uVec_r(k, i) = uVals[i][k + Layer<T>::out_size];
            uVec_c(k, i) = uVals[i][k + Layer<T>::out_size * 2];
        }
    }
}

template <typename T>
void GRULayer<T>::setUVals(T** uVals)
{
    for(int i = 0; i < Layer<T>::out_size; ++i)
    {
        for(int k = 0; k < Layer<T>::out_size; ++k)
        {
            uVec_z(k, i) = uVals[i][k];
            uVec_r(k, i) = uVals[i][k + Layer<T>::out_size];
            uVec_c(k, i) = uVals[i][k + Layer<T>::out_size * 2];
        }
    }
}

template <typename T>
void GRULayer<T>::setBVals(const std::vector<std::vector<T>>& bVals)
{
    for(int i = 0; i < 2; ++i)
    {
        for(int k = 0; k < Layer<T>::out_size; ++k)
        {
            bVec_z(k, i) = bVals[i][k];
            bVec_r(k, i) = bVals[i][k + Layer<T>::out_size];
            bVec_c(k, i) = bVals[i][k + Layer<T>::out_size * 2];
        }
    }
}

template <typename T>
void GRULayer<T>::setBVals(T** bVals)
{
    for(int i = 0; i < 2; ++i)
    {
        for(int k = 0; k < Layer<T>::out_size; ++k)
        {
            bVec_z(k, i) = bVals[i][k];
            bVec_r(k, i) = bVals[i][k + Layer<T>::out_size];
            bVec_c(k, i) = bVals[i][k + Layer<T>::out_size * 2];
        }
    }
}

template <typename T>
T GRULayer<T>::getWVal(int i, int k) const noexcept
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> set = wVec_z;
    if(k > 2 * Layer<T>::out_size)
        set = wVec_c;
    else if(k > Layer<T>::out_size)
        set = wVec_r;

    return set(k % Layer<T>::out_size, i);
}

template <typename T>
T GRULayer<T>::getUVal(int i, int k) const noexcept
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> set = uVec_z;
    if(k > 2 * Layer<T>::out_size)
        set = uVec_c;
    else if(k > Layer<T>::out_size)
        set = uVec_r;

    return set(k % Layer<T>::out_size, i);
}

template <typename T>
T GRULayer<T>::getBVal(int i, int k) const noexcept
{
    Eigen::Matrix<T, Eigen::Dynamic, 2> set = bVec_z;
    if(k > 2 * Layer<T>::out_size)
        set = bVec_c;
    else if(k > Layer<T>::out_size)
        set = bVec_r;

    return set(k % Layer<T>::out_size, i);
}

//====================================================
template <typename T, int in_sizet, int out_sizet, SampleRateCorrectionMode sampleRateCorr>
GRULayerT<T, in_sizet, out_sizet, sampleRateCorr>::GRULayerT()
    : outs(outs_internal)
{
    wVec_z = k_type::Zero();
    wVec_r = k_type::Zero();
    wVec_c = k_type::Zero();

    uVec_z = r_type::Zero();
    uVec_r = r_type::Zero();
    uVec_c = r_type::Zero();

    bVec_z = b_type::Zero();
    bVec_r = b_type::Zero();
    bVec_c0 = b_type::Zero();
    bVec_c1 = b_type::Zero();

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
