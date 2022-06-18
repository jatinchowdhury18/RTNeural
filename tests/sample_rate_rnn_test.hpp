#pragma once

#include <iostream>
#include <RTNeural.h>

auto getSampleRateVector(double sampleRate)
{
    static constexpr auto numSeconds = 0.25;
    const auto numSamples = int (numSeconds * sampleRate);

    std::vector<double> x;
    x.resize(numSamples, 0.0);
    for(int n = 0; n < numSamples; ++n)
        x[n] = std::sin(600.0 * (double) n / sampleRate);

    return x;
}

template <template <RTNeural::SampleRateCorrectionMode> class ModelType, RTNeural::SampleRateCorrectionMode mode, int RLayerIdx, typename MultType>
int runModelTest(const std::string& modelFile, MultType sampleRateMult)
{
    static constexpr auto baseSampleRate = 48000.0;

    ModelType<RTNeural::SampleRateCorrectionMode::None> baseSampleRateModel;
    std::ifstream jsonStream1("models/" + modelFile, std::ifstream::binary);
    baseSampleRateModel.parseJson(jsonStream1);
    baseSampleRateModel.reset();
    auto baseSampleRateSignal = getSampleRateVector(baseSampleRate);
    for(auto& sample : baseSampleRateSignal)
        sample = baseSampleRateModel.forward(&sample);

    ModelType<mode> testSampleRateModel;
    std::ifstream jsonStream2("models/" + modelFile, std::ifstream::binary);
    testSampleRateModel.parseJson(jsonStream2);
    testSampleRateModel.reset();
    testSampleRateModel.template get<RLayerIdx>().prepare(sampleRateMult);
    auto testSampleRateSignal = getSampleRateVector(baseSampleRate * sampleRateMult);
    for(auto& sample : testSampleRateSignal)
        sample = testSampleRateModel.forward(&sample);

    double maxErr = 0.0;
    const auto checkSamplesInc = int (sampleRateMult * 4.0);
    for (int i = 0, j = (int) std::ceil (sampleRateMult) - 1; i < baseSampleRateSignal.size() && j < testSampleRateSignal.size(); i += 4, j += checkSamplesInc)
        maxErr = std::max(maxErr, std::abs(baseSampleRateSignal[i] - testSampleRateSignal[j]));
//    std::cout << maxErr << std::endl;

    double maxErrLimit = sampleRateMult == std::floor (sampleRateMult) ? 0.0 : 5.0e-4;
    if (maxErr > maxErrLimit)
    {
        std::cout << "        FAIL! Max error: " << maxErr << std::endl;
        return 1;
    }

    return 0;
}

template <RTNeural::SampleRateCorrectionMode mode>
using GRUModel = RTNeural::ModelT<double, 1, 1,
    RTNeural::DenseT<double, 1, 8>,
    RTNeural::TanhActivationT<double, 8>,
    RTNeural::GRULayerT<double, 8, 8, mode>,
    RTNeural::DenseT<double, 8, 8>,
    RTNeural::SigmoidActivationT<double, 8>,
    RTNeural::DenseT<double, 8, 1>>;

template <RTNeural::SampleRateCorrectionMode mode>
using GRU1DModel = RTNeural::ModelT<double, 1, 1,
    RTNeural::GRULayerT<double, 1, 8, mode>,
    RTNeural::DenseT<double, 8, 8>,
    RTNeural::SigmoidActivationT<double, 8>,
    RTNeural::DenseT<double, 8, 1>>;

template <RTNeural::SampleRateCorrectionMode mode>
using LSTMModel = RTNeural::ModelT<double, 1, 1,
    RTNeural::DenseT<double, 1, 8>,
    RTNeural::TanhActivationT<double, 8>,
    RTNeural::LSTMLayerT<double, 8, 8, mode>,
    RTNeural::DenseT<double, 8, 1>>;

template <RTNeural::SampleRateCorrectionMode mode>
using LSTM1DModel = RTNeural::ModelT<double, 1, 1,
    RTNeural::LSTMLayerT<double, 1, 8, mode>,
    RTNeural::DenseT<double, 8, 1>>;

int runTestAtSampleRateOffset(const std::string& model)
{
    std::cout << "    Testing model " << model << std::endl;

    using namespace RTNeural;
    int result = 0;
    if(model == "gru")
    {
        result |= runModelTest<GRUModel, SampleRateCorrectionMode::NoInterp, 2>("gru.json", 3);
        result |= runModelTest<GRUModel, SampleRateCorrectionMode::LinInterp, 2>("gru.json", 1.75);
    }
    else if(model == "gru_1d")
    {
        result |= runModelTest<GRU1DModel, SampleRateCorrectionMode::NoInterp, 0>("gru_1d.json", 3);
        result |= runModelTest<GRU1DModel, SampleRateCorrectionMode::LinInterp, 0>("gru_1d.json", 1.75);
    }
    else if(model == "lstm")
    {
        result |= runModelTest<LSTMModel, SampleRateCorrectionMode::NoInterp, 2>("lstm.json", 4);
        result |= runModelTest<LSTMModel, SampleRateCorrectionMode::LinInterp, 2>("lstm.json", 2.5);
    }
    else if(model == "lstm_1d")
    {
        result |= runModelTest<LSTM1DModel, SampleRateCorrectionMode::NoInterp, 0>("lstm_1d.json", 2);
        result |= runModelTest<LSTM1DModel, SampleRateCorrectionMode::LinInterp, 0>("lstm_1d.json", 2.25);
    }

    return result;
}

int sampleRateRNNTest()
{
    std::cout << "Running Sample Rate RNN Tests..." << std::endl;

    int result = 0;
    for (auto layerType : { "gru", "gru_1d", "lstm", "lstm_1d" })
        result |= runTestAtSampleRateOffset(layerType);

    return result;
}
