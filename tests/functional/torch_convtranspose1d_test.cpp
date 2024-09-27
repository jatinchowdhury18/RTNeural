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
        testing::DoubleNear(static_cast<double>(actual), 2e-5));
}

template <typename T, int IN_SIZE, int OUT_SIZE, int KERNEL_SIZE, int PADDING, int DILATION, int GROUPS, int STRIDE>
void testTorchConvTranspose1DModel(const std::string& model_file_path,
    const std::string& model_input_file_path,
    const std::string& model_output_file_path)
{
    const auto model_file = std::string { RTNEURAL_ROOT_DIR } + model_file_path;
    std::ifstream jsonStream(std::string(model_file), std::ifstream::binary);
    nlohmann::json modelJson;
    jsonStream >> modelJson;

    RTNeural::Conv1D<T> model(IN_SIZE, OUT_SIZE, KERNEL_SIZE, DILATION, GROUPS);

    RTNeural::torch_helpers::loadConvTranspose1D<T>(modelJson, "", model);
    model.reset();
    std::ifstream modelInputsFile { std::string { RTNEURAL_ROOT_DIR } + model_input_file_path };
    const auto inputs = RTNeural::torch_helpers::detail::transpose(load_csv::loadFile2d<T>(modelInputsFile));
#if RTNEURAL_USE_XSIMD
    using Array = std::array<T, RTNeural::ceil_div(OUT_SIZE, (int) xsimd::batch<T>::size) * xsimd::batch<T>::size>;
    std::vector<Array, xsimd::aligned_allocator<Array>> outputs {};
#else
    std::vector<std::array<T, OUT_SIZE>> outputs {};
#endif
    const size_t out_base_size = (inputs.size() - 1) * STRIDE - PADDING + 1;
    static constexpr size_t tconv_side_padding = DILATION * (KERNEL_SIZE - 1) - PADDING;
    outputs.resize(out_base_size + tconv_side_padding, {});

    alignas(RTNEURAL_DEFAULT_ALIGNMENT) std::array<T, IN_SIZE> input_data = { 0 };
    alignas(RTNEURAL_DEFAULT_ALIGNMENT) std::array<T, IN_SIZE> zeroentry = { 0 };

    for(size_t i = 0; i < out_base_size + PADDING; ++i)
    {
        if(i < PADDING)
        {
            if((i % STRIDE) == 0)
            {
                std::copy(std::begin(inputs[i / STRIDE]), std::end(inputs[i / STRIDE]), std::begin(input_data));
                model.skip(input_data.data());
            }
            else // Feed zeroes to input
                model.skip(zeroentry.data());
        }
        else
        {
            if((i % STRIDE) == 0)
            {
                std::copy(std::begin(inputs[i / STRIDE]), std::end(inputs[i / STRIDE]), std::begin(input_data));
                model.forward(input_data.data(), outputs[i - PADDING].data());
            }
            else // Feed zeroes to input
                model.forward(zeroentry.data(), outputs[i - PADDING].data());
        }
        // std::cout << "Written at " << i-PADDING <<" " << outputs[i-PADDING][0] <<std::endl;
    }
    for(size_t i = 0; i < tconv_side_padding; ++i)
    {
        // Feed the same zeropad to the model
        model.forward(zeroentry.data(), outputs[out_base_size + i].data());
        // std::cout << "Written at " << out_base_size+i <<" " << outputs[out_base_size+i][0] <<std::endl;
    }
    std::ifstream modelOutputsFile { std::string { RTNEURAL_ROOT_DIR } + model_output_file_path };
    const auto expected_y = RTNeural::torch_helpers::detail::transpose(load_csv::loadFile2d<T>(modelOutputsFile));

    for(size_t n = 0; n < expected_y.size(); ++n)
    {
        for(size_t j = 0; j < (size_t) OUT_SIZE; ++j)
        {
            expectNear(outputs[n][j], expected_y[n][j]);
        }
    }
}

template <typename T, int IN_SIZE, int OUT_SIZE, int KERNEL_SIZE, int PADDING, int DILATION, int GROUPS, int STRIDE>
void testTorchConvTranspose1DModelComptime(const std::string& model_file_path,
    const std::string& model_input_file_path,
    const std::string& model_output_file_path)
{
    const auto model_file = std::string { RTNEURAL_ROOT_DIR } + model_file_path;
    std::ifstream jsonStream(std::string(model_file), std::ifstream::binary);
    nlohmann::json modelJson;
    jsonStream >> modelJson;

    RTNeural::Conv1DT<T, IN_SIZE, OUT_SIZE, KERNEL_SIZE, DILATION, GROUPS, true> model {};

    RTNeural::torch_helpers::loadConvTranspose1D<T>(modelJson, "", model);
    model.reset();
    std::ifstream modelInputsFile { std::string { RTNEURAL_ROOT_DIR } + model_input_file_path };
    const auto inputs = RTNeural::torch_helpers::detail::transpose(load_csv::loadFile2d<T>(modelInputsFile));
#if RTNEURAL_USE_XSIMD
    using Array = std::array<T, RTNeural::ceil_div(OUT_SIZE, (int) xsimd::batch<T>::size) * xsimd::batch<T>::size>;
    std::vector<Array, xsimd::aligned_allocator<Array>> outputs {};
#else
    std::vector<std::array<T, OUT_SIZE>> outputs {};
#endif
    const size_t out_base_size = (inputs.size() - 1) * STRIDE - PADDING + 1;
    static constexpr size_t tconv_side_padding = DILATION * (KERNEL_SIZE - 1) - PADDING;
    outputs.resize(out_base_size + tconv_side_padding, {});

    for(size_t i = 0; i < out_base_size + PADDING; ++i)
    {
#if RTNEURAL_USE_EIGEN
        alignas(RTNEURAL_DEFAULT_ALIGNMENT) Eigen::Matrix<T, IN_SIZE, 1> input_data {};
        input_data.setZero();
#elif RTNEURAL_USE_XSIMD
        alignas(RTNEURAL_DEFAULT_ALIGNMENT) xsimd::batch<T> input_data[RTNeural::ceil_div(IN_SIZE, (int)xsimd::batch<T>::size)] {};
#else
        alignas(RTNEURAL_DEFAULT_ALIGNMENT) T input_data[IN_SIZE] {};
#endif
        if(i < PADDING)
        {
            if((i % STRIDE) == 0)
            {
#if RTNEURAL_USE_XSIMD
                std::copy(std::begin(inputs[i / STRIDE]),
                    std::end(inputs[i / STRIDE]),
                    reinterpret_cast<T*>(std::begin(input_data)));
#else
                std::copy(std::begin(inputs[i / STRIDE]),
                    std::end(inputs[i / STRIDE]),
                    std::begin(input_data));
#endif
                model.skip(input_data);
            }
            else // Feed zeroes to input
            {
                // Stride in ConvTranspose1d does zero-stuffing
                model.skip(input_data);
            }
        }
        else
        {
            if((i % STRIDE) == 0)
            {
#if RTNEURAL_USE_XSIMD
                std::copy(std::begin(inputs[i / STRIDE]),
                    std::end(inputs[i / STRIDE]),
                    reinterpret_cast<T*>(std::begin(input_data)));
#else
                std::copy(std::begin(inputs[i / STRIDE]),
                    std::end(inputs[i / STRIDE]),
                    std::begin(input_data));
#endif
                model.forward(input_data);
#if RTNEURAL_USE_XSIMD
                std::copy(reinterpret_cast<T*>(std::begin(model.outs)),
                    reinterpret_cast<T*>(std::end(model.outs)),
                    std::begin(outputs[i - PADDING]));
#else
                std::copy(std::begin(model.outs),
                    std::end(model.outs),
                    std::begin(outputs[i - PADDING]));
#endif
            }
            else // Feed zeroes to input
            {
                model.forward(input_data);
#if RTNEURAL_USE_XSIMD
                std::copy(reinterpret_cast<T*>(std::begin(model.outs)),
                    reinterpret_cast<T*>(std::end(model.outs)),
                    std::begin(outputs[i - PADDING]));
#else
                std::copy(std::begin(model.outs),
                    std::end(model.outs),
                    std::begin(outputs[i - PADDING]));
#endif
            }
        }
    }
    for(size_t i = 0; i < tconv_side_padding; ++i)
    {
        // Feed the same zeropad to the model
#if RTNEURAL_USE_EIGEN
        alignas(RTNEURAL_DEFAULT_ALIGNMENT) Eigen::Matrix<T, IN_SIZE, 1> input_data {};
        input_data.setZero();
#elif RTNEURAL_USE_XSIMD
        alignas(RTNEURAL_DEFAULT_ALIGNMENT) xsimd::batch<T> input_data[RTNeural::ceil_div(IN_SIZE, (int)xsimd::batch<T>::size)] {};
#else
        alignas(RTNEURAL_DEFAULT_ALIGNMENT) T input_data[IN_SIZE] {};
#endif
        model.forward(input_data);
#if RTNEURAL_USE_XSIMD
        std::copy(reinterpret_cast<T*>(std::begin(model.outs)),
            reinterpret_cast<T*>(std::end(model.outs)),
            std::begin(outputs[out_base_size + i]));
#else
        std::copy(std::begin(model.outs),
            std::end(model.outs),
            std::begin(outputs[out_base_size + i]));
#endif
    }
    std::ifstream modelOutputsFile { std::string { RTNEURAL_ROOT_DIR } + model_output_file_path };
    const auto expected_y = RTNeural::torch_helpers::detail::transpose(load_csv::loadFile2d<T>(modelOutputsFile));

    for(size_t n = 0; n < expected_y.size(); ++n)
    {
        for(size_t j = 0; j < (size_t) OUT_SIZE; ++j)
        {
            expectNear(outputs[n][j], expected_y[n][j]);
        }
    }
}

template <typename T, int IN_SIZE, int OUT_SIZE, int KERNEL_SIZE, int PADDING, int DILATION, int GROUPS, int STRIDE>
void testStreamingTorchConvTranspose1DModel(const std::string& model_file_path,
    const std::string& model_input_file_path,
    const std::string& model_output_file_path)
{
    const auto model_file = std::string { RTNEURAL_ROOT_DIR } + model_file_path;
    std::ifstream jsonStream(std::string(model_file), std::ifstream::binary);
    nlohmann::json modelJson;
    jsonStream >> modelJson;

    RTNeural::Conv1D<T> model { IN_SIZE, OUT_SIZE, KERNEL_SIZE, DILATION, GROUPS };

    RTNeural::torch_helpers::loadConvTranspose1D<T>(modelJson, "", model);
    model.reset();
    std::ifstream modelInputsFile { std::string { RTNEURAL_ROOT_DIR } + model_input_file_path };
    const auto inputs = RTNeural::torch_helpers::detail::transpose(load_csv::loadFile2d<T>(modelInputsFile));
#if RTNEURAL_USE_XSIMD
    using Array = std::array<T, RTNeural::ceil_div(OUT_SIZE, (int) xsimd::batch<T>::size) * xsimd::batch<T>::size>;
    std::vector<Array, xsimd::aligned_allocator<Array>> outputs {};
#else
    std::vector<std::array<T, OUT_SIZE>> outputs {};
#endif
    const size_t out_base_size = (inputs.size() - 1) * STRIDE - PADDING + 1;
    static constexpr size_t tconv_side_padding = DILATION * (KERNEL_SIZE - 1) - PADDING;
    outputs.resize(out_base_size + tconv_side_padding, {});
    alignas(RTNEURAL_DEFAULT_ALIGNMENT) std::array<T, IN_SIZE> input_data = { 0 };
    alignas(RTNEURAL_DEFAULT_ALIGNMENT) std::array<T, IN_SIZE> zeroentry = { 0 };

    for(size_t i = 0; i < out_base_size; ++i)
    {
        if((i % STRIDE) == 0)
        {
            std::copy(std::begin(inputs[i / STRIDE]), std::end(inputs[i / STRIDE]), std::begin(input_data));
            model.forward(input_data.data(), outputs[i].data());
        }
        else
            // Stride in ConvTranspose1d does zero-stuffing
            model.forward(zeroentry.data(), outputs[i].data());
    }
    for(size_t i = 0; i < tconv_side_padding; ++i)
    {
        // Feed the same zeropad to the model
        model.forward(zeroentry.data(), outputs[out_base_size + i].data());
    }
    std::ifstream modelOutputsFile { std::string { RTNEURAL_ROOT_DIR } + model_output_file_path };
    const auto expected_y = RTNeural::torch_helpers::detail::transpose(load_csv::loadFile2d<T>(modelOutputsFile));

    for(size_t n = 0; n < expected_y.size(); ++n)
    {
        for(size_t j = 0; j < (size_t) OUT_SIZE; ++j)
        {
            expectNear(outputs[n][j], expected_y[n][j]);
        }
    }
}

template <typename T, int IN_SIZE, int OUT_SIZE, int KERNEL_SIZE, int PADDING, int DILATION, int GROUPS, int STRIDE>
void testStreamingTorchConvTranspose1DModelComptime(const std::string& model_file_path,
    const std::string& model_input_file_path,
    const std::string& model_output_file_path)
{
    const auto model_file = std::string { RTNEURAL_ROOT_DIR } + model_file_path;
    std::ifstream jsonStream(std::string(model_file), std::ifstream::binary);
    nlohmann::json modelJson;
    jsonStream >> modelJson;

    RTNeural::Conv1DT<T, IN_SIZE, OUT_SIZE, KERNEL_SIZE, DILATION, GROUPS> model {};

    RTNeural::torch_helpers::loadConvTranspose1D<T>(modelJson, "", model);
    model.reset();
    std::ifstream modelInputsFile { std::string { RTNEURAL_ROOT_DIR } + model_input_file_path };
    const auto inputs = RTNeural::torch_helpers::detail::transpose(load_csv::loadFile2d<T>(modelInputsFile));
#if RTNEURAL_USE_XSIMD
    using Array = std::array<T, RTNeural::ceil_div(OUT_SIZE, (int) xsimd::batch<T>::size) * xsimd::batch<T>::size>;
    std::vector<Array, xsimd::aligned_allocator<Array>> outputs {};
#else
    std::vector<std::array<T, OUT_SIZE>> outputs {};
#endif
    const size_t out_base_size = (inputs.size() - 1) * STRIDE - PADDING + 1;
    static constexpr size_t tconv_side_padding = DILATION * (KERNEL_SIZE - 1) - PADDING;
    outputs.resize(out_base_size + tconv_side_padding, {});

    for(size_t i = 0; i < out_base_size; ++i)
    {
#if RTNEURAL_USE_EIGEN
        alignas(RTNEURAL_DEFAULT_ALIGNMENT) Eigen::Matrix<T, IN_SIZE, 1> input_data {};
        input_data.setZero();
#elif RTNEURAL_USE_XSIMD
        alignas(RTNEURAL_DEFAULT_ALIGNMENT) xsimd::batch<T> input_data[RTNeural::ceil_div(IN_SIZE, (int)xsimd::batch<T>::size)] {};
#else
        alignas(RTNEURAL_DEFAULT_ALIGNMENT) T input_data[IN_SIZE] {};
#endif
        if((i % STRIDE) == 0)
        {
#if RTNEURAL_USE_XSIMD
            std::copy(std::begin(inputs[i / STRIDE]),
                std::end(inputs[i / STRIDE]),
                reinterpret_cast<T*>(std::begin(input_data)));
#else
            std::copy(std::begin(inputs[i / STRIDE]),
                std::end(inputs[i / STRIDE]),
                std::begin(input_data));
#endif
            model.forward(input_data);
#if RTNEURAL_USE_XSIMD
            std::copy(reinterpret_cast<T*>(std::begin(model.outs)),
                reinterpret_cast<T*>(std::end(model.outs)),
                std::begin(outputs[i]));
#else
            std::copy(std::begin(model.outs),
                std::end(model.outs),
                std::begin(outputs[i]));
#endif
        }
        else
        {
            // Stride in ConvTranspose1d does zero-stuffing
            model.forward(input_data);
#if RTNEURAL_USE_XSIMD
            std::copy(reinterpret_cast<T*>(std::begin(model.outs)),
                reinterpret_cast<T*>(std::end(model.outs)),
                std::begin(outputs[i]));
#else
            std::copy(std::begin(model.outs),
                std::end(model.outs),
                std::begin(outputs[i]));
#endif
        }
    }
    for(size_t i = 0; i < tconv_side_padding; ++i)
    {
        // Feed the same zeropad to the model
#if RTNEURAL_USE_EIGEN
        alignas(RTNEURAL_DEFAULT_ALIGNMENT) Eigen::Matrix<T, IN_SIZE, 1> input_data {};
        input_data.setZero();
#elif RTNEURAL_USE_XSIMD
        alignas(RTNEURAL_DEFAULT_ALIGNMENT) xsimd::batch<T> input_data[RTNeural::ceil_div(IN_SIZE, (int)xsimd::batch<T>::size)] {};
#else
        alignas(RTNEURAL_DEFAULT_ALIGNMENT) T input_data[IN_SIZE] {};
#endif
        model.forward(input_data);
#if RTNEURAL_USE_XSIMD
        std::copy(reinterpret_cast<T*>(std::begin(model.outs)),
            reinterpret_cast<T*>(std::end(model.outs)),
            std::begin(outputs[out_base_size + i]));
#else
        std::copy(std::begin(model.outs),
            std::end(model.outs),
            std::begin(outputs[out_base_size + i]));
#endif
    }
    std::ifstream modelOutputsFile { std::string { RTNEURAL_ROOT_DIR } + model_output_file_path };
    const auto expected_y = RTNeural::torch_helpers::detail::transpose(load_csv::loadFile2d<T>(modelOutputsFile));

    for(size_t n = 0; n < expected_y.size(); ++n)
    {
        for(size_t j = 0; j < (size_t) OUT_SIZE; ++j)
        {
            expectNear(outputs[n][j], expected_y[n][j]);
        }
    }
}
}

TEST(TestTorchConvTranspose1D, modelOutputMatchesPythonImplementationForFloatsRuntime)
{
    testTorchConvTranspose1DModel<float, 4, 15, 5, 3, 1, 1, 3>(
        "models/convtranspose1d_torch.json",
        "test_data/convtranspose1d_torch_x_python.csv",
        "test_data/convtranspose1d_torch_y_python.csv");
}

TEST(TestTorchConvTranspose1D, modelOutputMatchesPythonImplementationForFloatsComptime)
{
    testTorchConvTranspose1DModelComptime<float, 4, 15, 5, 3, 1, 1, 3>(
        "models/convtranspose1d_torch.json",
        "test_data/convtranspose1d_torch_x_python.csv",
        "test_data/convtranspose1d_torch_y_python.csv");
}

TEST(TestTorchConvTranspose1D, modelOutputMatchesPythonImplementationForDoublesRuntime)
{
    testTorchConvTranspose1DModel<double, 4, 15, 5, 3, 1, 1, 3>(
        "models/convtranspose1d_torch.json",
        "test_data/convtranspose1d_torch_x_python.csv",
        "test_data/convtranspose1d_torch_y_python.csv");
}

TEST(TestTorchConvTranspose1D, modelOutputMatchesPythonImplementationForDoublesComptime)
{
    testTorchConvTranspose1DModelComptime<double, 4, 15, 5, 3, 1, 1, 3>(
        "models/convtranspose1d_torch.json",
        "test_data/convtranspose1d_torch_x_python.csv",
        "test_data/convtranspose1d_torch_y_python.csv");
}

TEST(TestTorchConvTranspose1D, streaming_modelOutputMatchesPythonImplementationForFloatsRuntime)
{
    testStreamingTorchConvTranspose1DModel<float, 4, 15, 5, 3, 1, 1, 3>(
        "models/convtranspose1d_torch.json",
        "test_data/convtranspose1d_torch_x_python_cc.csv",
        "test_data/convtranspose1d_torch_y_python_cc.csv");
}

TEST(TestTorchConvTranspose1D, streaming_modelOutputMatchesPythonImplementationForFloatsComptime)
{
    testStreamingTorchConvTranspose1DModelComptime<float, 4, 15, 5, 3, 1, 1, 3>(
        "models/convtranspose1d_torch.json",
        "test_data/convtranspose1d_torch_x_python_cc.csv",
        "test_data/convtranspose1d_torch_y_python_cc.csv");
}

TEST(TestTorchConvTranspose1D, streaming_modelOutputMatchesPythonImplementationForDoublesRuntime)
{
    testStreamingTorchConvTranspose1DModel<double, 4, 15, 5, 3, 1, 1, 3>(
        "models/convtranspose1d_torch.json",
        "test_data/convtranspose1d_torch_x_python_cc.csv",
        "test_data/convtranspose1d_torch_y_python_cc.csv");
}

TEST(TestTorchConvTranspose1D, streaming_modelOutputMatchesPythonImplementationForDoublesComptime)
{
    testStreamingTorchConvTranspose1DModelComptime<double, 4, 15, 5, 3, 1, 1, 3>(
        "models/convtranspose1d_torch.json",
        "test_data/convtranspose1d_torch_x_python_cc.csv",
        "test_data/convtranspose1d_torch_y_python_cc.csv");
}
