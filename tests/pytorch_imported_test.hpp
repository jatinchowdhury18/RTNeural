#pragma once

#include <RTNeural.h>
#include "load_csv.hpp"

namespace pytorch_imported_test
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

int pytorch_imported_test()
{
    std::cout << "TESTING PYTORCH IMPORTED MODEL..." << std::endl;

    const std::string model_file = "models/pytorch_imported.json";
    const std::string data_file = "test_data/pytorch_x.csv";
    const std::string ref_data_file = "test_data/pytorch_y.csv";
    constexpr double threshold = 1.0e-12;

    std::ifstream pythonX(data_file);
    auto xData = load_csv::loadFile<TestType>(pythonX);

    std::ifstream pythonYRef(ref_data_file);
    auto yRefData = load_csv::loadFile<TestType>(pythonYRef);

    std::vector<TestType> yData(xData.size(), (TestType)0);
    {
        std::cout << "Loading non-templated model" << std::endl;
        std::ifstream jsonStream(model_file, std::ifstream::binary);
        auto model = RTNeural::json_parser::parseJson<TestType>(jsonStream, true);
        processModel(*model.get(), xData, yData);
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

    std::cout << "SUCCESS" << std::endl;
    return 0;
}

} // namespace pytorch_imported_test
