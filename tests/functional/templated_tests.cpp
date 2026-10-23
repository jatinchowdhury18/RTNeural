#include <gmock/gmock.h>

#include "load_csv.hpp"
#include "test_configs.hpp"
#include "test_maths_provider.hpp"

namespace
{
using TestType = double;
using namespace RTNeural;

template <typename T, typename StaticModelType>
void runTestTemplated(const TestConfig& test)
{
    std::ifstream jsonStream(std::string { RTNEURAL_ROOT_DIR } + test.model_file, std::ifstream::binary);
    StaticModelType static_model;
    static_model.parseJson(jsonStream, true);
    static_model.reset();

    std::ifstream pythonX(std::string { RTNEURAL_ROOT_DIR } + test.x_data_file);
    auto xData = load_csv::loadFile<T>(pythonX);

    std::ifstream pythonY(std::string { RTNEURAL_ROOT_DIR } + test.y_data_file);
    const auto yRefData = load_csv::loadFile<T>(pythonY);

    jsonStream.seekg(0);
    auto dynamic_model = RTNeural::json_parser::parseJson<T>(jsonStream, true);
    dynamic_model->reset();

    std::vector<T> yDataStatic(xData.size(), (T)0);
    std::vector<T> yDataDynamic(xData.size(), (T)0);
    for(size_t n = 0; n < xData.size(); ++n)
    {
        T input[] = { xData[n] };
        yDataStatic[n] = static_model.forward(input);
        yDataDynamic[n] = dynamic_model->forward(input);
    }

    using namespace testing;
    EXPECT_THAT(yDataStatic, Pointwise(DoubleNear(test.threshold), yRefData));
    EXPECT_THAT(yDataDynamic, Pointwise(DoubleNear(test.threshold), yRefData));
}
}
#if RTNEURAL_USE_XSIMD
TEST(TestTemplatedModels, denseTHandlesNonAlignedInputSize)
{
    constexpr int inSize = 3;
    constexpr int outSize = 5;

    using Batch = xsimd::simd_type<TestType>;

    constexpr int simdWidth = static_cast<int>(Batch::size);
    constexpr int vInSize = (inSize + simdWidth - 1) / simdWidth;
    constexpr int vOutSize = (outSize + simdWidth - 1) / simdWidth;

    RTNeural::DenseT<TestType, inSize, outSize> dense;

    const std::vector<std::vector<TestType>> weights {
        { 1.0,  2.0,  3.0 },
        { 4.0,  5.0,  6.0 },
        { 7.0,  8.0,  9.0 },
        { 10.0, 11.0, 12.0 },
        { 13.0, 14.0, 15.0 }
    };

    const TestType biases[outSize] = {
        0.5, -0.5, 1.0, 2.0, -1.0
    };

    dense.setWeights(weights);
    dense.setBias(biases);

    // Pad the input buffer to a complete SIMD batch.
    alignas(RTNEURAL_DEFAULT_ALIGNMENT)
        TestType inputPadded[vInSize * simdWidth] = {
            2.0, -1.0, 3.0
        };

    Batch input[vInSize];

    for(int i = 0; i < vInSize; ++i)
        input[i] = xsimd::load_aligned(
            inputPadded + i * simdWidth
        );

    dense.forward(input);

    alignas(RTNEURAL_DEFAULT_ALIGNMENT)
        TestType actual[vOutSize * simdWidth] {};

    for(int i = 0; i < vOutSize; ++i)
        dense.outs[i].store_aligned(
            actual + i * simdWidth
        );

    const TestType expected[outSize] = {
        9.5, 20.5, 34.0, 47.0, 56.0
    };

    for(int i = 0; i < outSize; ++i)
        EXPECT_NEAR(actual[i], expected[i], 1.0e-12);
}

TEST(TestTemplatedModels, gruTHandlesNonAlignedDimensions)
{
    constexpr int inSize = 3;
    constexpr int hiddenSize = 3;

    using Batch = xsimd::simd_type<TestType>;
    constexpr int simdWidth = static_cast<int>(Batch::size);
    constexpr int vInSize = (inSize + simdWidth - 1) / simdWidth;
    constexpr int vOutSize = (hiddenSize + simdWidth - 1) / simdWidth;

    GRULayerT<TestType, inSize, hiddenSize> gru;

    // Layout: [input row][update gate, reset gate, candidate gate].
    std::vector<std::vector<TestType>> w(inSize,
        std::vector<TestType>(3 * hiddenSize));
    std::vector<std::vector<TestType>> u(hiddenSize,
        std::vector<TestType>(3 * hiddenSize));
    std::vector<std::vector<TestType>> b(2,
        std::vector<TestType>(3 * hiddenSize));

    for(int i = 0; i < inSize; ++i)
        for(int j = 0; j < 3 * hiddenSize; ++j)
            w[i][j] = 0.025 * (1 + i + j);

    for(int i = 0; i < hiddenSize; ++i)
        for(int j = 0; j < 3 * hiddenSize; ++j)
            u[i][j] = 0.015 * (1 + 2 * i - j);

    for(int j = 0; j < 3 * hiddenSize; ++j)
    {
        b[0][j] = 0.01 * (j + 1);
        b[1][j] = -0.005 * (j + 1);
    }

    gru.setWVals(w);
    gru.setUVals(u);
    gru.setBVals(b);
    gru.reset();

    const auto sigmoid = [](TestType x) {
        return 1.0 / (1.0 + std::exp(-x));
    };

    TestType state[hiddenSize] = {};
    const TestType samples[2][inSize] = {
        { 0.4, -0.7, 0.2 },
        { -0.3, 0.6, 0.8 }
    };

    for(int step = 0; step < 2; ++step)
    {
        TestType z[hiddenSize], r[hiddenSize], candidate[hiddenSize];
        TestType next[hiddenSize];

        for(int i = 0; i < hiddenSize; ++i)
        {
            TestType wz = 0, wr = 0, wh = 0;
            TestType uz = 0, ur = 0, uh = 0;

            for(int j = 0; j < inSize; ++j)
            {
                wz += samples[step][j] * w[j][i];
                wr += samples[step][j] * w[j][hiddenSize + i];
                wh += samples[step][j] * w[j][2 * hiddenSize + i];
            }

            for(int j = 0; j < hiddenSize; ++j)
            {
                uz += state[j] * u[j][i];
                ur += state[j] * u[j][hiddenSize + i];
                uh += state[j] * u[j][2 * hiddenSize + i];
            }

            z[i] = sigmoid(wz + uz + b[0][i] + b[1][i]);
            r[i] = sigmoid(wr + ur + b[0][hiddenSize + i]
                                      + b[1][hiddenSize + i]);

            candidate[i] = std::tanh(
                wh + b[0][2 * hiddenSize + i]
                + r[i] * (uh + b[1][2 * hiddenSize + i]));
        }

        for(int i = 0; i < hiddenSize; ++i)
            next[i] = z[i] * state[i] + (1.0 - z[i]) * candidate[i];

        alignas(RTNEURAL_DEFAULT_ALIGNMENT)
            TestType paddedInput[vInSize * simdWidth] = {};

        for(int i = 0; i < vInSize * simdWidth; ++i)
            paddedInput[i] = 1.25 + 0.1 * i;

        for(int i = 0; i < inSize; ++i)
            paddedInput[i] = samples[step][i];

        Batch input[vInSize];
        for(int i = 0; i < vInSize; ++i)
            input[i] = xsimd::load_aligned(
                paddedInput + i * simdWidth);

        alignas(RTNEURAL_DEFAULT_ALIGNMENT)
            TestType paddedState[vOutSize * simdWidth] = {};

        for(int i = 0; i < hiddenSize; ++i)
            paddedState[i] = state[i];

        for(int i = hiddenSize; i < vOutSize * simdWidth; ++i)
            paddedState[i] = 1.25 + 0.1 * i;

        for(int i = 0; i < vOutSize; ++i)
            gru.outs[i] = xsimd::load_aligned(
                paddedState + i * simdWidth);

        gru.forward(input);

        alignas(RTNEURAL_DEFAULT_ALIGNMENT)
            TestType actual[vOutSize * simdWidth] = {};

        for(int i = 0; i < vOutSize; ++i)
            gru.outs[i].store_aligned(actual + i * simdWidth);

        for(int i = 0; i < hiddenSize; ++i)
            EXPECT_NEAR(actual[i], next[i], 1.0e-10)
                << "step=" << step << ", hidden=" << i;

        for(int i = 0; i < hiddenSize; ++i)
            state[i] = next[i];
    }
}

TEST(TestTemplatedModels, lstmTHandlesNonAlignedDimensions)
{
    constexpr int inSize = 3;
    constexpr int hiddenSize = 3;

    using Batch = xsimd::simd_type<TestType>;
    constexpr int simdWidth = static_cast<int>(Batch::size);
    constexpr int vInSize = (inSize + simdWidth - 1) / simdWidth;
    constexpr int vOutSize = (hiddenSize + simdWidth - 1) / simdWidth;

    LSTMLayerT<TestType, inSize, hiddenSize> lstm;

    // Gate layout follows the templated LSTM setters:
    // input, forget, candidate, output.
    std::vector<std::vector<TestType>> w(
        inSize, std::vector<TestType>(4 * hiddenSize));
    std::vector<std::vector<TestType>> u(
        hiddenSize, std::vector<TestType>(4 * hiddenSize));
    std::vector<TestType> b(4 * hiddenSize);

    for(int i = 0; i < inSize; ++i)
        for(int j = 0; j < 4 * hiddenSize; ++j)
            w[i][j] = 0.02 * (1 + i + j);

    for(int i = 0; i < hiddenSize; ++i)
        for(int j = 0; j < 4 * hiddenSize; ++j)
            u[i][j] = 0.012 * (1 + 2 * i - j);

    for(int j = 0; j < 4 * hiddenSize; ++j)
        b[j] = 0.003 * (j + 1) - 0.015;

    lstm.setWVals(w);
    lstm.setUVals(u);
    lstm.setBVals(b);
    lstm.reset();

    const auto sigmoid = [](TestType x) {
        return 1.0 / (1.0 + std::exp(-x));
    };

    TestType hidden[hiddenSize] = {};
    TestType cell[hiddenSize] = {};

    const TestType samples[2][inSize] = {
        { 0.4, -0.7, 0.2 },
        { -0.3, 0.6, 0.8 }
    };

    for(int step = 0; step < 2; ++step)
    {
        TestType nextHidden[hiddenSize];
        TestType nextCell[hiddenSize];

        for(int i = 0; i < hiddenSize; ++i)
        {
            TestType wi = 0, wf = 0, wc = 0, wo = 0;
            TestType ui = 0, uf = 0, uc = 0, uo = 0;

            // Input-to-gate contributions.
            for(int j = 0; j < inSize; ++j)
            {
                wi += samples[step][j] * w[j][i];
                wf += samples[step][j] * w[j][hiddenSize + i];
                wc += samples[step][j] * w[j][2 * hiddenSize + i];
                wo += samples[step][j] * w[j][3 * hiddenSize + i];
            }

            // Previous hidden-state contributions.
            for(int j = 0; j < hiddenSize; ++j)
            {
                ui += hidden[j] * u[j][i];
                uf += hidden[j] * u[j][hiddenSize + i];
                uc += hidden[j] * u[j][2 * hiddenSize + i];
                uo += hidden[j] * u[j][3 * hiddenSize + i];
            }

            const TestType inputGate =
                sigmoid(wi + ui + b[i]);

            const TestType forgetGate =
                sigmoid(wf + uf + b[hiddenSize + i]);

            const TestType candidate =
                std::tanh(wc + uc + b[2 * hiddenSize + i]);

            const TestType outputGate =
                sigmoid(wo + uo + b[3 * hiddenSize + i]);

            nextCell[i] =
                forgetGate * cell[i] + inputGate * candidate;

            nextHidden[i] =
                outputGate * std::tanh(nextCell[i]);
        }

        // Supply a padded, aligned input buffer to the SIMD layer.
        alignas(RTNEURAL_DEFAULT_ALIGNMENT)
            TestType paddedInput[vInSize * simdWidth] = {};

        for(int i = 0; i < vInSize * simdWidth; ++i)
            paddedInput[i] = 1.25 + 0.1 * i;

        for(int i = 0; i < inSize; ++i)
            paddedInput[i] = samples[step][i];

        Batch input[vInSize];

        for(int i = 0; i < vInSize; ++i)
            input[i] = xsimd::load_aligned(
                paddedInput + i * simdWidth);

        alignas(RTNEURAL_DEFAULT_ALIGNMENT)
            TestType paddedState[vOutSize * simdWidth] = {};

        for(int i = 0; i < hiddenSize; ++i)
            paddedState[i] = hidden[i];

        for(int i = hiddenSize; i < vOutSize * simdWidth; ++i)
            paddedState[i] = 1.25 + 0.1 * i;

        for(int i = 0; i < vOutSize; ++i)
            lstm.outs[i] = xsimd::load_aligned(
                paddedState + i * simdWidth);

        lstm.forward(input);

        alignas(RTNEURAL_DEFAULT_ALIGNMENT)
            TestType actual[vOutSize * simdWidth] = {};

        for(int i = 0; i < vOutSize; ++i)
            lstm.outs[i].store_aligned(actual + i * simdWidth);

        for(int i = 0; i < hiddenSize; ++i)
            EXPECT_NEAR(actual[i], nextHidden[i], 1.0e-10)
                << "step=" << step << ", hidden=" << i;

        for(int i = 0; i < hiddenSize; ++i)
        {
            hidden[i] = nextHidden[i];
            cell[i] = nextCell[i];
        }
    }
}

#endif

TEST(TestTemplatedModels, modelOutputMatchesPythonImplementationForDense)
{
    using ModelType = ModelT<TestType, 1, 1,
        DenseT<TestType, 1, 8>,
        TanhActivationT<TestType, 8>,
        DenseT<TestType, 8, 8>,
        ReLuActivationT<TestType, 8>,
        DenseT<TestType, 8, 8>,
        ELuActivationT<TestType, 8>,
        DenseT<TestType, 8, 8>,
        SoftmaxActivationT<TestType, 8>,
        DenseT<TestType, 8, 1>,
        DenseT<TestType, 1, 8, false>,
        DenseT<TestType, 8, 8, false>,
        DenseT<TestType, 8, 1, false>>;

    runTestTemplated<TestType, ModelType>(tests.at("dense"));
}

TEST(TestTemplatedModels, modelOutputMatchesPythonImplementationForDenseWithMathsProvider)
{
    using ModelType = ModelT<TestType, 1, 1,
        DenseT<TestType, 1, 8>,
        TanhActivationT<TestType, 8, TestMathsProvider>,
        DenseT<TestType, 8, 8>,
        ReLuActivationT<TestType, 8>,
        DenseT<TestType, 8, 8>,
        ELuActivationT<TestType, 8, 1, 1, TestMathsProvider>,
        DenseT<TestType, 8, 8>,
        SoftmaxActivationT<TestType, 8, TestMathsProvider>,
        DenseT<TestType, 8, 1>,
        DenseT<TestType, 1, 8, false>,
        DenseT<TestType, 8, 8, false>,
        DenseT<TestType, 8, 1, false>>;

    runTestTemplated<TestType, ModelType>(tests.at("dense"));
}

TEST(TestTemplatedModels, modelOutputMatchesPythonImplementationForConv1D)
{
    using ModelType = ModelT<TestType, 1, 1,
        DenseT<TestType, 1, 8>,
        TanhActivationT<TestType, 8>,
        Conv1DT<TestType, 8, 4, 3, 1, true>,
        TanhActivationT<TestType, 4>,
        BatchNorm1DT<TestType, 4>,
        PReLUActivationT<TestType, 4>,
        Conv1DT<TestType, 4, 4, 1, 1>,
        TanhActivationT<TestType, 4>,
        Conv1DT<TestType, 4, 6, 3, 2, 2>,
        TanhActivationT<TestType, 6>,
        BatchNorm1DT<TestType, 6, false>,
        PReLUActivationT<TestType, 6>,
        DenseT<TestType, 6, 1>,
        SigmoidActivationT<TestType, 1>>;

    runTestTemplated<TestType, ModelType>(tests.at("conv1d"));
}

TEST(TestTemplatedModels, modelOutputMatchesPythonImplementationForGRU)
{
    using ModelType = ModelT<TestType, 1, 1,
        DenseT<TestType, 1, 8>,
        TanhActivationT<TestType, 8>,
        GRULayerT<TestType, 8, 8>,
        DenseT<TestType, 8, 8>,
        SigmoidActivationT<TestType, 8>,
        DenseT<TestType, 8, 1>>;

    runTestTemplated<TestType, ModelType>(tests.at("gru"));
}

TEST(TestTemplatedModels, modelOutputMatchesPythonImplementationForGRUWithMathsProvider)
{
    using ModelType = ModelT<TestType, 1, 1,
        DenseT<TestType, 1, 8>,
        TanhActivationT<TestType, 8, TestMathsProvider>,
        GRULayerT<TestType, 8, 8, RTNeural::SampleRateCorrectionMode::None, TestMathsProvider>,
        DenseT<TestType, 8, 8>,
        SigmoidActivationT<TestType, 8, TestMathsProvider>,
        DenseT<TestType, 8, 1>>;

    runTestTemplated<TestType, ModelType>(tests.at("gru"));
}

TEST(TestTemplatedModels, modelOutputMatchesPythonImplementationForGRU1D)
{
    using ModelType = ModelT<TestType, 1, 1,
        GRULayerT<TestType, 1, 8>,
        DenseT<TestType, 8, 8>,
        SigmoidActivationT<TestType, 8>,
        DenseT<TestType, 8, 1>>;

    runTestTemplated<TestType, ModelType>(tests.at("gru_1d"));
}

TEST(TestTemplatedModels, modelOutputMatchesPythonImplementationForGRU1DWithMathsProvider)
{
    using ModelType = ModelT<TestType, 1, 1,
        GRULayerT<TestType, 1, 8, RTNeural::SampleRateCorrectionMode::None, TestMathsProvider>,
        DenseT<TestType, 8, 8>,
        SigmoidActivationT<TestType, 8, TestMathsProvider>,
        DenseT<TestType, 8, 1>>;

    runTestTemplated<TestType, ModelType>(tests.at("gru_1d"));
}

TEST(TestTemplatedModels, modelOutputMatchesPythonImplementationForLSTM)
{
    using ModelType = ModelT<TestType, 1, 1,
        DenseT<TestType, 1, 8>,
        TanhActivationT<TestType, 8>,
        LSTMLayerT<TestType, 8, 8>,
        DenseT<TestType, 8, 1>>;

    runTestTemplated<TestType, ModelType>(tests.at("lstm"));
}

TEST(TestTemplatedModels, modelOutputMatchesPythonImplementationForLSTMWithMathsProvider)
{
    using ModelType = ModelT<TestType, 1, 1,
        DenseT<TestType, 1, 8>,
        TanhActivationT<TestType, 8, TestMathsProvider>,
        LSTMLayerT<TestType, 8, 8, RTNeural::SampleRateCorrectionMode::None, TestMathsProvider>,
        DenseT<TestType, 8, 1>>;

    runTestTemplated<TestType, ModelType>(tests.at("lstm"));
}

TEST(TestTemplatedModels, modelOutputMatchesPythonImplementationForLSTM1D)
{
    using ModelType = ModelT<TestType, 1, 1,
        LSTMLayerT<TestType, 1, 8>,
        DenseT<TestType, 8, 1>>;

    runTestTemplated<TestType, ModelType>(tests.at("lstm_1d"));
}

TEST(TestTemplatedModels, modelOutputMatchesPythonImplementationForLSTM1DWithMathsProvider)
{
    using ModelType = ModelT<TestType, 1, 1,
        LSTMLayerT<TestType, 1, 8, RTNeural::SampleRateCorrectionMode::None, TestMathsProvider>,
        DenseT<TestType, 8, 1>>;

    runTestTemplated<TestType, ModelType>(tests.at("lstm_1d"));
}
