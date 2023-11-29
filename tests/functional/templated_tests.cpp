#include <gmock/gmock.h>

#include "load_csv.hpp"
#include "test_configs.hpp"
#include <RTNeural/RTNeural.h>

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
        DenseT<TestType, 8, 1>>;

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

TEST(TestTemplatedModels, modelOutputMatchesPythonImplementationForGRU1D)
{
    using ModelType = ModelT<TestType, 1, 1,
        GRULayerT<TestType, 1, 8>,
        DenseT<TestType, 8, 8>,
        SigmoidActivationT<TestType, 8>,
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

TEST(TestTemplatedModels, modelOutputMatchesPythonImplementationForLSTM1D)
{
    using ModelType = ModelT<TestType, 1, 1,
        LSTMLayerT<TestType, 1, 8>,
        DenseT<TestType, 8, 1>>;

    runTestTemplated<TestType, ModelType>(tests.at("lstm_1d"));
}
