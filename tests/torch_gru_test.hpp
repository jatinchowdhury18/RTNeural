#pragma once

#include "load_csv.hpp"
#include "RTNeural/RTNeural.h"

namespace torch_gru_test
{
template <typename T>
int testTorchGRUModel()
{
    using ModelType = RTNeural::ModelT<T, 1, 1, RTNeural::GRULayerT<T, 1, 8>, RTNeural::DenseT<T, 8, 1>>;

    const auto loadModel = [](std::ifstream& jsonStream, ModelType& model)
    {
        nlohmann::json modelJson;
        jsonStream >> modelJson;

        auto& gru = model.template get<0>();
        RTNeural::torch_helpers::loadGRU<T> (modelJson, "gru.", gru);

        auto& dense = model.template get<1>();
        RTNeural::torch_helpers::loadDense<T> (modelJson, "dense.", dense);
    };

    if (std::is_same<T, float>::value)
        std::cout << "TESTING TORCH/GRU MODEL WITH DATA TYPE: FLOAT" << std::endl;
    else
        std::cout << "TESTING TORCH/GRU MODEL WITH DATA TYPE: DOUBLE" << std::endl;
    std::ifstream jsonStream("models/gru_torch.json", std::ifstream::binary);

    ModelType model;
    loadModel(jsonStream, model);
    model.reset();

    std::ifstream modelInputsFile { "test_data/gru_torch_x_python.csv" };
    const auto inputs = load_csv::loadFile<T>(modelInputsFile);
    std::vector<T> outputs {};
    outputs.resize(inputs.size(), {});

    for(size_t i = 0; i < inputs.size(); ++i)
    {
        outputs[i] = model.forward(&inputs[i]);
    }

    std::ifstream modelOutputsFile { "test_data/gru_torch_y_python.csv" };
    const auto expected_y = load_csv::loadFile<T>(modelOutputsFile);

    size_t nErrs = 0;
    T max_error = (T)0;
    for(size_t n = 0; n < inputs.size(); ++n)
    {
        auto err = std::abs(outputs[n] - expected_y[n]);
        if(err > (T) 1.0e-6)
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

int torchGRUTest()
{
    int result = 0;
    result |= torch_gru_test::testTorchGRUModel<float>();
    result |= torch_gru_test::testTorchGRUModel<double>();
    return result;
}
