#include <gmock/gmock.h>

#include "load_csv.hpp"
#include <RTNeural/RTNeural.h>

#if __cplusplus > 202002L
#include <stdfloat>
#endif

namespace
{
template <typename T>
void expectNear(T const& expected, T const& actual, double error_thresh)
{
    EXPECT_THAT(
        static_cast<double>(expected),
        testing::DoubleNear(static_cast<double>(actual), error_thresh));
}

template <typename T>
void testTorchConv1DModel(double error_thresh = 1e-6)
{
    const auto model_file = std::string { RTNEURAL_ROOT_DIR } + "models/conv1d_torch.json";
    std::ifstream jsonStream(model_file, std::ifstream::binary);
    nlohmann::json modelJson;
    jsonStream >> modelJson;

    RTNeural::ModelT<T, 1, 12, RTNeural::Conv1DT<T, 1, 12, 5, 1>> model;
    RTNeural::torch_helpers::loadConv1D<T>(modelJson, "", model.template get<0>());
    model.reset();

    std::ifstream modelInputsFile { std::string { RTNEURAL_ROOT_DIR } + "test_data/conv1d_torch_x_python.csv" };
    const auto inputs = load_csv::loadFile<T>(modelInputsFile);
    std::vector<std::array<T, 12>> outputs {};
    outputs.resize(inputs.size(), {});

    for(size_t i = 0; i < inputs.size(); ++i)
    {
        model.forward(&inputs[i]);
        std::copy(model.getOutputs(), model.getOutputs() + 12, outputs[i].begin());
    }

    std::ifstream modelOutputsFile { std::string { RTNEURAL_ROOT_DIR } + "test_data/conv1d_torch_y_python.csv" };
    const auto expected_y = RTNeural::torch_helpers::detail::transpose(load_csv::loadFile2d<T>(modelOutputsFile));

    for(size_t n = 0; n < expected_y.size(); ++n)
    {
        for(size_t j = 0; j < outputs[n].size(); ++j)
        {
            expectNear(outputs[n + 4][j], expected_y[n][j], error_thresh);
        }
    }
}
}

TEST(TestTorchConv1D, modelOutputMatchesPythonImplementationForFloats)
{
    testTorchConv1DModel<float>();
}

TEST(TestTorchConv1D, modelOutputMatchesPythonImplementationForDoubles)
{
    testTorchConv1DModel<double>();
}

#if __STDCPP_FLOAT16_T__
TEST(TestTorchConv1D, modelOutputMatchesPythonImplementationForFloat16)
{
    testTorchConv1DModel<std::float16_t>(1.0e-3);
}
#endif

#if __STDCPP_BFLOAT16_T__
TEST(TestTorchConv1D, modelOutputMatchesPythonImplementationForBFloat16)
{
    testTorchConv1DModel<std::bfloat16_t>(1.0e-2);
}
#endif
