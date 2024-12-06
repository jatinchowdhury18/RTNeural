#include <gmock/gmock.h>

#include "load_csv.hpp"
#include <RTNeural/RTNeural.h>

#if __cplusplus > 202002L
#include <stdfloat>
#endif

namespace
{
template <typename T>
void testTorchLSTMModel(double error_thresh = 1.0e-6)
{
    using ModelType = RTNeural::ModelT<T, 1, 1,
        RTNeural::LSTMLayerT<T, 1, 8>,
        RTNeural::LSTMLayerT<T, 8, 8>,
        RTNeural::LSTMLayerT<T, 8, 8>,
        RTNeural::LSTMLayerT<T, 8, 8>,
        RTNeural::DenseT<T, 8, 1>>;

    const auto loadModel = [](std::ifstream& jsonStream, ModelType& model)
    {
        nlohmann::json modelJson;
        jsonStream >> modelJson;

        auto& lstm = model.template get<0>();
        RTNeural::torch_helpers::loadLSTM<T>(modelJson, "lstm.", lstm);

        auto& lstm2_l0 = model.template get<1>();
        RTNeural::torch_helpers::loadLSTM<T>(modelJson, "lstm2.", lstm2_l0, true, 0);
        auto& lstm2_l1 = model.template get<2>();
        RTNeural::torch_helpers::loadLSTM<T>(modelJson, "lstm2.", lstm2_l1, true, 1);
        auto& lstm2_l2 = model.template get<3>();
        RTNeural::torch_helpers::loadLSTM<T>(modelJson, "lstm2.", lstm2_l2, true, 2);

        auto& dense = model.template get<4>();
        RTNeural::torch_helpers::loadDense<T>(modelJson, "dense.", dense);
    };

    const auto model_file = std::string { RTNEURAL_ROOT_DIR } + "models/lstm_torch.json";
    std::ifstream jsonStream(model_file, std::ifstream::binary);

    ModelType model;
    loadModel(jsonStream, model);
    model.reset();

    std::ifstream modelInputsFile { std::string { RTNEURAL_ROOT_DIR } + "test_data/lstm_torch_x_python.csv" };
    const auto inputs = load_csv::loadFile<T>(modelInputsFile);
    std::vector<T> outputs {};
    outputs.resize(inputs.size(), {});

    for(size_t i = 0; i < inputs.size(); ++i)
    {
        outputs[i] = model.forward(&inputs[i]);
    }

    std::ifstream modelOutputsFile { std::string { RTNEURAL_ROOT_DIR } + "test_data/lstm_torch_y_python.csv" };
    const auto expected_y = load_csv::loadFile<T>(modelOutputsFile);

    using namespace testing;
    EXPECT_THAT(outputs, Pointwise(DoubleNear(error_thresh), expected_y));
}
}

TEST(TestTorchLSTM, modelOutputMatchesPythonImplementationForFloats)
{
    testTorchLSTMModel<float>();
}

TEST(TestTorchLSTM, modelOutputMatchesPythonImplementationForDoubles)
{
    testTorchLSTMModel<double>();
}

#if __STDCPP_FLOAT16_T__
TEST(TestTorchLSTM, modelOutputMatchesPythonImplementationForFloat16)
{
    testTorchLSTMModel<std::float16_t>(1.0e-3);
}
#endif

#if __STDCPP_BFLOAT16_T__
TEST(TestTorchLSTM, modelOutputMatchesPythonImplementationForBFloat16)
{
    testTorchLSTMModel<std::bfloat16_t>(1.0e-2);
}
#endif
