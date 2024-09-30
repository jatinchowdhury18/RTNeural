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
    static constexpr size_t STRIDE = 3, KS = 5, OUT_CH = 12;

    // Use dynamic model.
    RTNeural::StridedConv1D<T> model(1, OUT_CH, KS, 1, STRIDE, 1);
    RTNeural::torch_helpers::loadConv1D<T>(modelJson, "", model);
    model.reset();

    std::ifstream modelInputsFile { std::string { RTNEURAL_ROOT_DIR } + "test_data/conv1d_torch_x_python_stride_3.csv" };
    const auto inputs = load_csv::loadFile<T>(modelInputsFile);
#if RTNEURAL_USE_XSIMD
    using Array = std::array<T, RTNeural::ceil_div(OUT_CH, xsimd::batch<T>::size) * xsimd::batch<T>::size>;
    std::vector<Array, xsimd::aligned_allocator<Array>> outputs {};
#else
    std::vector<std::array<T, OUT_CH>> outputs {};
#endif
    const size_t start_point = KS - 1;
    outputs.resize((inputs.size() - start_point) / STRIDE, {});
    // std::cout << "Out size " << outputs.size() << "\n";

    for(size_t i = 0; i < start_point; ++i)
        model.skip(&inputs[i]);

    for(size_t i = start_point; i < inputs.size(); ++i)
    {
        const auto out_idx = (i - start_point) / STRIDE;
        model.forward(&inputs[i], outputs[out_idx].data());
    }

    std::ifstream modelOutputsFile { std::string { RTNEURAL_ROOT_DIR } + "test_data/conv1d_torch_y_python_stride_3.csv" };
    const auto expected_y = RTNeural::torch_helpers::detail::transpose(load_csv::loadFile2d<T>(modelOutputsFile));

    for(size_t n = 0; n < expected_y.size(); ++n)
    {
        for(size_t j = 0; j < OUT_CH; ++j)
        {
            expectNear(outputs[n][j], expected_y[n][j]);
        }
    }
}

template <typename T>
void testTorchConv1DModelComptime()
{
    const auto model_file = std::string { RTNEURAL_ROOT_DIR } + "models/conv1d_torch_stride_3.json";
    std::ifstream jsonStream(model_file, std::ifstream::binary);
    nlohmann::json modelJson;
    jsonStream >> modelJson;
    static constexpr size_t STRIDE = 3, KS = 5, OUT_CH = 12;

    RTNeural::StridedConv1DT<T, 1, OUT_CH, KS, 1, STRIDE> model;
    RTNeural::torch_helpers::loadConv1D<T>(modelJson, "", model);
    model.reset();

    std::ifstream modelInputsFile { std::string { RTNEURAL_ROOT_DIR } + "test_data/conv1d_torch_x_python_stride_3.csv" };
    const auto inputs = load_csv::loadFile<T>(modelInputsFile);
#if RTNEURAL_USE_XSIMD
    using Array = std::array<T, RTNeural::ceil_div(OUT_CH, xsimd::batch<T>::size) * xsimd::batch<T>::size>;
    std::vector<Array, xsimd::aligned_allocator<Array>> outputs {};
#else
    std::vector<std::array<T, OUT_CH>> outputs {};
#endif
    const size_t start_point = KS - 1;
    outputs.resize((inputs.size() - start_point) / STRIDE, {});
    // std::cout << "Out size " << outputs.size() << "\n";

#if RTNEURAL_USE_EIGEN
    alignas(RTNEURAL_DEFAULT_ALIGNMENT) Eigen::Matrix<T, 1, 1> input_data {};
    input_data.setZero();
#elif RTNEURAL_USE_XSIMD
    alignas(RTNEURAL_DEFAULT_ALIGNMENT) xsimd::batch<T> input_data[RTNeural::ceil_div(1, (int)xsimd::batch<T>::size)] {};
#else
    alignas(RTNEURAL_DEFAULT_ALIGNMENT) T input_data[1] {};
#endif

    for(size_t i = 0; i < start_point; ++i)
    {
        input_data[0] = inputs[i];
        model.skip(input_data);
    }

    for(size_t i = start_point; i < inputs.size(); ++i)
    {
        input_data[0] = inputs[i];
        model.forward(input_data);

        const auto out_idx = (i - start_point) / STRIDE;
#if RTNEURAL_USE_XSIMD
        int batch_idx = 0;
        for (auto& batch : model.outs)
        {
            batch.store_aligned (outputs[out_idx].data() + batch_idx);
            batch_idx += xsimd::batch<T>::size;
        }
#else
        std::copy(std::begin(model.outs),
            std::end(model.outs),
            std::begin(outputs[out_idx]));
#endif
    }

    std::ifstream modelOutputsFile { std::string { RTNEURAL_ROOT_DIR } + "test_data/conv1d_torch_y_python_stride_3.csv" };
    const auto expected_y = RTNeural::torch_helpers::detail::transpose(load_csv::loadFile2d<T>(modelOutputsFile));

    for(size_t n = 0; n < expected_y.size(); ++n)
    {
        for(size_t j = 0; j < OUT_CH; ++j)
        {
            expectNear(outputs[n][j], expected_y[n][j]);
        }
    }
}
}

TEST(TestTorchConv1DStride, modelOutputMatchesPythonImplementationForFloatsRuntime)
{
    testTorchConv1DModel<float>();
}

TEST(TestTorchConv1DStride, modelOutputMatchesPythonImplementationForFloatsComptime)
{
    testTorchConv1DModelComptime<float>();
}

TEST(TestTorchConv1DStride, modelOutputMatchesPythonImplementationForDoublesRuntime)
{
    testTorchConv1DModel<double>();
}

TEST(TestTorchConv1DStride, modelOutputMatchesPythonImplementationForDoublesComptime)
{
    testTorchConv1DModelComptime<double>();
}
