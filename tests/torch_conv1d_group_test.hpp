#pragma once

#include "load_csv.hpp"
#include "RTNeural/RTNeural.h"

namespace torch_conv1d_group_test
{
template <typename T>
std::vector<std::vector<T>> loadFile2D(std::ifstream& stream)
{
    std::vector<std::vector<T>> vec;

    std::string line;
    if(stream.is_open()) {
        while(std::getline(stream, line)) {
            std::vector<T> lineVec;
            std::string num;
            for (auto ch : line) {
                if (ch == ',') {
                    lineVec.push_back(static_cast<T>(std::stod(num)));
                    num.clear();
                    continue;
                }

                num.push_back(ch);
            }

            lineVec.push_back(static_cast<T>(std::stod(num)));
            vec.push_back(lineVec);
        }

        stream.close();
    }

    return RTNeural::torch_helpers::detail::transpose(vec);
}

template <typename T>
int testTorchConv1DGroupModel()
{
    if (std::is_same<T, float>::value)
        std::cout << "TESTING TORCH/CONV1D GROUP MODEL WITH DATA TYPE: FLOAT" << std::endl;
    else
        std::cout << "TESTING TORCH/CONV1D GROUP MODEL WITH DATA TYPE: DOUBLE" << std::endl;

    const auto model_file = std::string { RTNEURAL_ROOT_DIR } + "models/conv1d_torch_group.json";
    std::ifstream jsonStream(model_file, std::ifstream::binary);

    nlohmann::json modelJson;
    jsonStream >> modelJson;

    RTNeural::ModelT<T, 6, 3, RTNeural::Conv1DT<T, 6, 3, 1, 1, 3, false>> model;
    RTNeural::torch_helpers::loadConv1D<T>(modelJson, "", model.template get<0>());
    model.reset();

    std::ifstream modelInputsFile { std::string { RTNEURAL_ROOT_DIR } + "test_data/conv1d_torch_group_x_python.csv" };
    const auto inputs = load_csv::loadFile2d<T, 6>(modelInputsFile);
    std::vector<std::array<T, 3>> outputs {};
    outputs.resize(inputs.size(), {});

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        model.forward(inputs[i].data());
        std::copy(model.getOutputs(), model.getOutputs() + 3, outputs[i].begin());
    }

    std::ifstream modelOutputsFile { std::string { RTNEURAL_ROOT_DIR } + "test_data/conv1d_torch_group_y_python.csv" };
    const auto expected_y = loadFile2D<T> (modelOutputsFile);

    size_t nErrs = 0;
    T max_error = (T)0;
    for(size_t n = 0; n < expected_y.size(); ++n)
    {
        for(size_t j = 0; j < outputs[n].size(); ++j)
        {
            auto err = std::abs(outputs[n][j] - expected_y[n][j]);
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
    result |= torch_conv1d_group_test::testTorchConv1DGroupModel<float>();
    result |= torch_conv1d_group_test::testTorchConv1DGroupModel<double>();
    return result;
}
