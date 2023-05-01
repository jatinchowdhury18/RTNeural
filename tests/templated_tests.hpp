#pragma once

#include "load_csv.hpp"
#include "test_configs.hpp"
#include <iostream>

template <typename T, typename ModelType>
int runTestTemplated(const TestConfig& test)
{
    std::cout << "TESTING " << test.name << " TEMPLATED IMPLEMENTATION..." << std::endl;

    std::ifstream jsonStream(test.model_file, std::ifstream::binary);
    ModelType model;
    model.parseJson(jsonStream, true);
    model.reset();

    std::ifstream pythonX(test.x_data_file);
    auto xData = load_csv::loadFile<T>(pythonX);

    std::ifstream pythonY(test.y_data_file);
    const auto yRefData = load_csv::loadFile<T>(pythonY);

    std::vector<T> yData(xData.size(), (T)0);
    for(size_t n = 0; n < xData.size(); ++n)
    {
        T input[] = { xData[n] };
        yData[n] = model.forward(input);
    }

    size_t nErrs = 0;
    T max_error = (T)0;
    for(size_t n = 0; n < xData.size(); ++n)
    {
        auto err = std::abs(yData[n] - yRefData[n]);
        if(err > test.threshold)
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

int templatedTests(std::string arg)
{
    using namespace RTNeural;
    using TestType = double;

#if MODELT_AVAILABLE
    int result = 0;

    if(arg == "dense")
    {
        using ModelType = ModelT<TestType, 1, 1,
            DenseT<TestType, 1, 8>,
            TanhActivationT<TestType, 8>,
            DenseT<TestType, 8, 8>,
            ReLuActivationT<TestType, 8>,
            DenseT<TestType, 8, 8>,
            ELuActivationT<TestType, 8>,
            DenseT<TestType, 8, 8>,
            SoftmaxActivationT<TestType, 8>,
            DenseT<TestType, 8, 1>>;
        result |= runTestTemplated<TestType, ModelType>(tests.at(arg));
    }
    else if(arg == "conv1d")
    {
        using ModelType = ModelT<TestType, 1, 1,
            DenseT<TestType, 1, 8>,
            TanhActivationT<TestType, 8>,
            Conv1DT<TestType, 8, 4, 3, 1, true>,
            TanhActivationT<TestType, 4>,
            BatchNorm1DT<TestType, 4>,
            PReLUActivationT<TestType, 4>,
            Conv1DT<TestType, 4, 4, 1, 1>,
            TanhActivationT<TestType, 4>,
            Conv1DT<TestType, 4, 4, 3, 2>,
            TanhActivationT<TestType, 4>,
            BatchNorm1DT<TestType, 4, false>,
            PReLUActivationT<TestType, 4>,
            DenseT<TestType, 4, 1>,
            SigmoidActivationT<TestType, 1>>;
        result |= runTestTemplated<TestType, ModelType>(tests.at(arg));
    }
    else if(arg == "gru")
    {
        using ModelType = ModelT<TestType, 1, 1,
            DenseT<TestType, 1, 8>,
            TanhActivationT<TestType, 8>,
            GRULayerT<TestType, 8, 8>,
            DenseT<TestType, 8, 8>,
            SigmoidActivationT<TestType, 8>,
            DenseT<TestType, 8, 1>>;
        result |= runTestTemplated<TestType, ModelType>(tests.at(arg));
    }
    else if(arg == "gru_1d")
    {
        using ModelType = ModelT<TestType, 1, 1,
            GRULayerT<TestType, 1, 8>,
            DenseT<TestType, 8, 8>,
            SigmoidActivationT<TestType, 8>,
            DenseT<TestType, 8, 1>>;
        result |= runTestTemplated<TestType, ModelType>(tests.at(arg));
    }
    else if(arg == "lstm")
    {
        using ModelType = ModelT<TestType, 1, 1,
            DenseT<TestType, 1, 8>,
            TanhActivationT<TestType, 8>,
            LSTMLayerT<TestType, 8, 8>,
            DenseT<TestType, 8, 1>>;
        result |= runTestTemplated<TestType, ModelType>(tests.at(arg));
    }
    else if(arg == "lstm_1d")
    {
        using ModelType = ModelT<TestType, 1, 1,
            LSTMLayerT<TestType, 1, 8>,
            DenseT<TestType, 8, 1>>;
        result |= runTestTemplated<TestType, ModelType>(tests.at(arg));
    }

    return result;

#else // @TODO
    return 0;
#endif // MODELT_AVAILABLE
}
