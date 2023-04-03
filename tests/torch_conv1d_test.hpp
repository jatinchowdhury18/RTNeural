#pragma once

#include "load_csv.hpp"
#include "RTNeural/RTNeural.h"

namespace torch_conv1d_test
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
int testTorchConv1DModel()
{
    if (std::is_same<T, float>::value)
        std::cout << "TESTING TORCH/CONV1D MODEL WITH DATA TYPE: FLOAT" << std::endl;
    else
        std::cout << "TESTING TORCH/CONV1D MODEL WITH DATA TYPE: DOUBLE" << std::endl;
    std::ifstream jsonStream("models/conv1d_torch.json", std::ifstream::binary);
    nlohmann::json modelJson;
    jsonStream >> modelJson;

    RTNeural::ModelT<T, 1, 12, RTNeural::Conv1DT<T, 1, 12, 5, 1>> model;
    RTNeural::torch_helpers::loadConv1D<T> (modelJson, "", model.template get<0>());
    model.reset();

    std::ifstream modelInputsFile { "test_data/conv1d_torch_x_python.csv" };
    const auto inputs = load_csv::loadFile<T>(modelInputsFile);
    std::vector<std::array<T, 12>> outputs {};
    outputs.resize(inputs.size(), {});

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        model.forward(&inputs[i]);
        std::copy(model.getOutputs(), model.getOutputs() + 12, outputs[i].begin());
    }

    std::ifstream modelOutputsFile { "test_data/conv1d_torch_y_python.csv" };
    const auto expected_y = loadFile2D<T> (modelOutputsFile);

    size_t nErrs = 0;
    T max_error = (T)0;
    for(size_t n = 0; n < expected_y.size(); ++n)
    {
        for(size_t j = 0; j < outputs[n].size(); ++j)
        {
            auto err = std::abs(outputs[n+4][j] - expected_y[n][j]);
            if(err > (T)1.0e-6)
            {
                max_error = std::max(err, max_error);
                nErrs++;

                // For debugging purposes
                // std::cout << "ERR: " << err << ", idx: " << n << std::endl;
                // std::cout << yData[n] << std::endl;
                // std::cout << yRefData[n] << std::endl;
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

int torchConv1DTest()
{
    int result = 0;
    result |= torch_conv1d_test::testTorchConv1DModel<float>();
    result |= torch_conv1d_test::testTorchConv1DModel<double>();
    return result;
}
