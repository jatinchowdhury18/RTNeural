#pragma once

#include <RTNeural.h>
#include "load_csv.hpp"

namespace model_test
{

using TestType = double;

template<typename ModelType>
void processModel(ModelType& model, const std::vector<TestType>& xData, std::vector<TestType>& yData)
{
    model.reset();
    for(size_t n = 0; n < xData.size(); ++n)
    {
        TestType input alignas(RTNEURAL_DEFAULT_ALIGNMENT)[] = { xData[n] };
        yData[n] = model.forward(input);
    }
}

int model_test()
{
    std::cout << "TESTING FULL MODEL..." << std::endl;

    const std::string model_file = "models/full_model.json";
    const std::string data_file = "test_data/dense_x_python.csv";
    constexpr double threshold = 1.0e-12;

    std::ifstream pythonX(data_file);
    auto xData = load_csv::loadFile<TestType>(pythonX);

    // non-templated model
    std::vector<TestType> yRefData(xData.size(), (TestType)0);
    {
        std::cout << "Loading non-templated model" << std::endl;
        std::ifstream jsonStream(model_file, std::ifstream::binary);
        auto modelRef = RTNeural::json_parser::parseJson<TestType>(jsonStream, true);
        processModel(*modelRef.get(), xData, yRefData);
    }

#if MODELT_AVAILABLE
    // templated model
    std::vector<TestType> yData(xData.size(), (TestType)0);
    {
        std::cout << "Loading templated model" << std::endl;
        RTNeural::ModelT<TestType, 1, 1,
            RTNeural::DenseT<TestType, 1, 8>,
            RTNeural::TanhActivationT<TestType, 8>,
            RTNeural::Conv1DT<TestType, 8, 4, 3, 2>,
            RTNeural::TanhActivationT<TestType, 4>,
            RTNeural::GRULayerT<TestType, 4, 8>,
            RTNeural::DenseT<TestType, 8, 1>
        > modelT;
        std::ifstream jsonStream(model_file, std::ifstream::binary);
        modelT.parseJson(jsonStream, true);
        processModel(modelT, xData, yData);
    }

    size_t nErrs = 0;
    TestType max_error = (TestType)0;
    for(size_t n = 0; n < xData.size(); ++n)
    {
        auto err = std::abs(yData[n] - yRefData[n]);
        if(err > threshold)
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
#endif

    std::cout << "SUCCESS" << std::endl;
    return 0;
}

} // namespace model_test
