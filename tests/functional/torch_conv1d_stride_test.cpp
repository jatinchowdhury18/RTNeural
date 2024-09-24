#include <gmock/gmock.h>

#include "load_csv.hpp"
#include <RTNeural/RTNeural.h>

namespace
{
template <typename T>
void expectNear(T const& expected, T const& actual)
{
    EXPECT_THAT(
        static_cast<double>(expected),
        testing::DoubleNear(static_cast<double>(actual), 1e-6));
}

template <typename T>
void testTorchConv1DModel()
{
    const auto model_file = std::string { RTNEURAL_ROOT_DIR } + "models/conv1d_torch_stride_3.json";
    std::ifstream jsonStream(model_file, std::ifstream::binary);
    nlohmann::json modelJson;
    jsonStream >> modelJson;
    const size_t STRIDE = 3, KS = 5, OUT_CH = 12;
    
    // Use dynamic model. Call model.skip() for striding.
    RTNeural::Conv1D<T> model(1, OUT_CH, KS, 1, 1);
    RTNeural::torch_helpers::loadConv1D<T>(modelJson, "", model);
    model.reset();

    std::ifstream modelInputsFile { std::string { RTNEURAL_ROOT_DIR } + "test_data/conv1d_torch_x_python_stride_3.csv" };
    const auto inputs = load_csv::loadFile<T>(modelInputsFile);
    std::vector<std::array<T, OUT_CH>> outputs {};
    const size_t start_point = KS-1;
    outputs.resize((inputs.size() - start_point)/ STRIDE, {});
    //std::cout << "Out size " << outputs.size() << "\n";
    
    for(size_t i = 0; i < start_point; ++i)
        model.skip(&inputs[i]);

    for(size_t i = start_point; i < inputs.size(); ++i)
    {
        if(((i-start_point) % STRIDE) == 0)
            model.forward(&inputs[i],outputs[(i-start_point)/STRIDE].data());
        else
            model.skip(&inputs[i]);
    }

    std::ifstream modelOutputsFile { std::string { RTNEURAL_ROOT_DIR } + "test_data/conv1d_torch_y_python_stride_3.csv" };
    const auto expected_y = RTNeural::torch_helpers::detail::transpose(load_csv::loadFile2d<T>(modelOutputsFile));

    for(size_t n = 0; n < expected_y.size(); ++n)
    {
        for(size_t j = 0; j < outputs[n].size(); ++j)
        {
            expectNear(outputs[n][j], expected_y[n][j]);
        }
    }
}
}

TEST(TestTorchConv1DStride, modelOutputMatchesPythonImplementationForFloats)
{
    testTorchConv1DModel<float>();
}

TEST(TestTorchConv1DStride, modelOutputMatchesPythonImplementationForDoubles)
{
    testTorchConv1DModel<double>();
}
