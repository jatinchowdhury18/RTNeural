#pragma once

#include "load_csv.hpp"
#include "RTNeural/RTNeural.h"

namespace torch_conv1d_group_test
{
int computeCrop(int input_size, int kernel_size, int dilation_rate)
{
    int output_size = (input_size  - dilation_rate * (kernel_size - 1) - 1) + 1;
    return input_size - output_size;
}

template <typename T, int input_size, int output_size, int kernel_size, int dilation_rate, int groups_of>
int testTorchConv1DGroupModel()
{
    if (std::is_same<T, float>::value)
        std::cout << "TESTING TORCH/CONV1D GROUP MODEL WITH DATA TYPE: FLOAT" << std::endl;
    else
        std::cout << "TESTING TORCH/CONV1D GROUP MODEL WITH DATA TYPE: DOUBLE" << std::endl;

    const auto model_file = 
        std::string { RTNEURAL_ROOT_DIR } + 
        "models/conv1d_torch_group_" +
        std::to_string(input_size) + "_" +
        std::to_string(output_size) + "_" +
        std::to_string(kernel_size) + "_" +
        std::to_string(dilation_rate) + "_" +
        std::to_string(groups_of) + ".json";
    std::ifstream jsonStream(model_file, std::ifstream::binary);

    nlohmann::json modelJson;
    jsonStream >> modelJson;

    RTNeural::ModelT<T, input_size, output_size, RTNeural::Conv1DT<T, input_size, output_size, kernel_size, dilation_rate, groups_of, false>> model;
    RTNeural::torch_helpers::loadConv1D<T>(modelJson, "", model.template get<0>());
    model.reset();

    std::ifstream modelInputsFile { std::string { RTNEURAL_ROOT_DIR } + "test_data/conv1d_torch_group_x_python.csv" };
    const auto inputs = load_csv::loadFile2d<T>(modelInputsFile);
    std::vector<std::array<T, output_size>> outputs {};
    outputs.resize(inputs.size(), {});

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        alignas (16) T input_copy[input_size + 4] {};
        std::copy (inputs[i].begin(), inputs[i].end(), std::begin (input_copy));
        model.forward(input_copy);
        std::copy(model.getOutputs(), model.getOutputs() + output_size, outputs[i].begin());
    }

    std::ifstream modelOutputsFile { 
        std::string { RTNEURAL_ROOT_DIR } + 
        "test_data/conv1d_torch_group_y_python_" +
            std::to_string(input_size) + "_" +
            std::to_string(output_size) + "_" +
            std::to_string(kernel_size) + "_" +
            std::to_string(dilation_rate) + "_" +
            std::to_string(groups_of) + ".csv"
        };
    const auto expected_y = RTNeural::torch_helpers::detail::transpose(load_csv::loadFile2d<T> (modelOutputsFile));

    int crop = computeCrop(static_cast<int>(inputs.size()), kernel_size, dilation_rate);

    size_t nErrs = 0;
    T max_error = (T)0;
    for(size_t n = 0; n < expected_y.size(); ++n)
    {
        for(size_t j = 0; j < outputs[crop + n].size(); ++j)
        {
            auto err = std::abs(outputs[crop + n][j] - expected_y[n][j]);
            if(err > (T)1.0e-6)
            {
                max_error = std::max(err, max_error);
                nErrs++;

                // For debugging purposes
                // std::cout << "ERR: " << err << ", idx: " << n << std::endl;
                // std::cout << "Output: " << outputs[n][j] << std::endl;
                // std::cout << "Expected: " << expected_y[n][j] << std::endl;
                // break;
            }
        }
    }

    if(nErrs > 0)
    {
        std::cout << "FAIL: " << nErrs << " errors!" << std::endl;
        std::cout << "Maximum error: " << max_error << std::endl;
        return 1;
    }

    std::cout << "SUCCESS" << std::endl;
    return 0;
}
}

int torchConv1DGroupTest()
{
    int result = 0;
    result |= torch_conv1d_group_test::testTorchConv1DGroupModel<float, 6, 3, 3, 1, 3>();
    result |= torch_conv1d_group_test::testTorchConv1DGroupModel<double, 6, 3, 3, 1, 3>();
    result |= torch_conv1d_group_test::testTorchConv1DGroupModel<float, 6, 3, 4, 10, 3>();
    result |= torch_conv1d_group_test::testTorchConv1DGroupModel<double, 6, 3, 4, 10, 3>();
    result |= torch_conv1d_group_test::testTorchConv1DGroupModel<float, 6, 6, 1, 1, 6>();
    result |= torch_conv1d_group_test::testTorchConv1DGroupModel<double, 6, 6, 1, 1, 6>();
    return result;
}
