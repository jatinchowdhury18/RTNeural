#include "lstm_eigen.h"

namespace RTNeural
{

template <typename T>
LSTMLayer<T>::LSTMLayer(int in_size, int out_size)
    : Layer<T>(in_size, out_size)
{
    Wf = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, in_size);
    Wi = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, in_size);
    Wo = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, in_size);
    Wc = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, in_size);

    Uf = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, out_size);
    Ui = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, out_size);
    Uo = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, out_size);
    Uc = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, out_size);

    bf = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, 1);
    bi = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, 1);
    bo = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, 1);
    bc = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, 1);

    fVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, 1);
    iVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, 1);
    oVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, 1);
    ctVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, 1);
    cVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, 1);

    inVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, 1);
    ht1 = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, 1);
    ct1 = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, 1);
}

template <typename T>
LSTMLayer<T>::LSTMLayer(std::initializer_list<int> sizes)
    : LSTMLayer<T>(*sizes.begin(), *(sizes.begin() + 1))
{
}

template <typename T>
LSTMLayer<T>::LSTMLayer(const LSTMLayer& other)
    : LSTMLayer<T>(other.in_size, other.out_size)
{
}

template <typename T>
LSTMLayer<T>& LSTMLayer<T>::operator=(const LSTMLayer<T>& other)
{
    return *this = LSTMLayer<T>(other);
}

template <typename T>
void LSTMLayer<T>::reset()
{
    std::fill(ht1.data(), ht1.data() + Layer<T>::out_size, (T)0);
    std::fill(ct1.data(), ct1.data() + Layer<T>::out_size, (T)0);
}

template <typename T>
void LSTMLayer<T>::setWVals(const std::vector<std::vector<T>>& wVals)
{
    for(int i = 0; i < Layer<T>::in_size; ++i)
    {
        for(int k = 0; k < Layer<T>::out_size; ++k)
        {
            Wi(k, i) = wVals[i][k];
            Wf(k, i) = wVals[i][k + Layer<T>::out_size];
            Wc(k, i) = wVals[i][k + Layer<T>::out_size * 2];
            Wo(k, i) = wVals[i][k + Layer<T>::out_size * 3];
        }
    }
}

template <typename T>
void LSTMLayer<T>::setUVals(const std::vector<std::vector<T>>& uVals)
{
    for(int i = 0; i < Layer<T>::out_size; ++i)
    {
        for(int k = 0; k < Layer<T>::out_size; ++k)
        {
            Ui(k, i) = uVals[i][k];
            Uf(k, i) = uVals[i][k + Layer<T>::out_size];
            Uc(k, i) = uVals[i][k + Layer<T>::out_size * 2];
            Uo(k, i) = uVals[i][k + Layer<T>::out_size * 3];
        }
    }
}

template <typename T>
void LSTMLayer<T>::setBVals(const std::vector<T>& bVals)
{
    for(int k = 0; k < Layer<T>::out_size; ++k)
    {
        bi(k) = bVals[k];
        bf(k) = bVals[k + Layer<T>::out_size];
        bc(k) = bVals[k + Layer<T>::out_size * 2];
        bo(k) = bVals[k + Layer<T>::out_size * 3];
    }
}

//====================================================
template <typename T, int in_sizet, int out_sizet, SampleRateCorrectionMode sampleRateCorr>
LSTMLayerT<T, in_sizet, out_sizet, sampleRateCorr>::LSTMLayerT()
    : outs(outs_internal)
{
    Wf = k_type::Zero();
    Wi = k_type::Zero();
    Wo = k_type::Zero();
    Wc = k_type::Zero();

    Uf = r_type::Zero();
    Ui = r_type::Zero();
    Uo = r_type::Zero();
    Uc = r_type::Zero();

    bf = b_type::Zero();
    bi = b_type::Zero();
    bo = b_type::Zero();
    bc = b_type::Zero();

    reset();
}

template <typename T, int in_sizet, int out_sizet, SampleRateCorrectionMode sampleRateCorr>
template <SampleRateCorrectionMode srCorr>
std::enable_if_t<srCorr == SampleRateCorrectionMode::NoInterp, void>
LSTMLayerT<T, in_sizet, out_sizet, sampleRateCorr>::prepare(int delaySamples)
{
    delayWriteIdx = delaySamples - 1;
    ct_delayed.resize(delayWriteIdx + 1, {});
    outs_delayed.resize(delayWriteIdx + 1, {});

    reset();
}

template <typename T, int in_sizet, int out_sizet, SampleRateCorrectionMode sampleRateCorr>
template <SampleRateCorrectionMode srCorr>
std::enable_if_t<srCorr == SampleRateCorrectionMode::LinInterp, void>
LSTMLayerT<T, in_sizet, out_sizet, sampleRateCorr>::prepare(T delaySamples)
{
    const auto delayOffFactor = delaySamples - std::floor(delaySamples);
    delayMult = (T)1 - delayOffFactor;
    delayPlus1Mult = delayOffFactor;

    delayWriteIdx = (int)std::ceil(delaySamples) - (int)std::ceil(delayOffFactor);
    ct_delayed.resize(delayWriteIdx + 1, {});
    outs_delayed.resize(delayWriteIdx + 1, {});

    reset();
}

template <typename T, int in_sizet, int out_sizet, SampleRateCorrectionMode sampleRateCorr>
void LSTMLayerT<T, in_sizet, out_sizet, sampleRateCorr>::reset()
{
    if(sampleRateCorr != SampleRateCorrectionMode::None)
    {
        for(auto& x : ct_delayed)
            x = out_type::Zero();

        for(auto& x : outs_delayed)
            x = out_type::Zero();
    }

    // reset output state
    outs = out_type::Zero();
    cVec = out_type::Zero();
}

// kernel weights
template <typename T, int in_sizet, int out_sizet, SampleRateCorrectionMode sampleRateCorr>
void LSTMLayerT<T, in_sizet, out_sizet, sampleRateCorr>::setWVals(const std::vector<std::vector<T>>& wVals)
{
    for(int i = 0; i < in_size; ++i)
    {
        for(int k = 0; k < out_size; ++k)
        {
            Wi(k, i) = wVals[i][k];
            Wf(k, i) = wVals[i][k + out_size];
            Wc(k, i) = wVals[i][k + out_size * 2];
            Wo(k, i) = wVals[i][k + out_size * 3];
        }
    }
}

// recurrent weights
template <typename T, int in_sizet, int out_sizet, SampleRateCorrectionMode sampleRateCorr>
void LSTMLayerT<T, in_sizet, out_sizet, sampleRateCorr>::setUVals(const std::vector<std::vector<T>>& uVals)
{
    for(int i = 0; i < out_size; ++i)
    {
        for(int k = 0; k < out_size; ++k)
        {
            Ui(k, i) = uVals[i][k];
            Uf(k, i) = uVals[i][k + out_size];
            Uc(k, i) = uVals[i][k + out_size * 2];
            Uo(k, i) = uVals[i][k + out_size * 3];
        }
    }
}

// biases
template <typename T, int in_sizet, int out_sizet, SampleRateCorrectionMode sampleRateCorr>
void LSTMLayerT<T, in_sizet, out_sizet, sampleRateCorr>::setBVals(const std::vector<T>& bVals)
{
    for(int k = 0; k < out_size; ++k)
    {
        bi(k) = bVals[k];
        bf(k) = bVals[k + out_size];
        bc(k) = bVals[k + out_size * 2];
        bo(k) = bVals[k + out_size * 3];
    }
}

} // namespace RTNeural
