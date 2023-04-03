#include "approx_tests.hpp"
#include "bad_model_test.hpp"
#include "conv2d_model.h"
#include "load_csv.hpp"
#include "model_test.hpp"
#include "sample_rate_rnn_test.hpp"
#include "templated_tests.hpp"
#include "test_configs.hpp"
#include "torch_conv1d_test.hpp"
#include "torch_gru_test.hpp"
#include "torch_lstm_test.hpp"
#include "util_tests.hpp"

// @TODO: make tests for both float and double precision
void help()
{
    std::cout << "RTNeural test suite:" << std::endl;
    std::cout << "Usage: rtneural_tests <test_type>" << std::endl;
    std::cout << std::endl;
    std::cout << "Available test types are:" << std::endl;

    std::cout << "    all" << std::endl;
    std::cout << "    util" << std::endl;
    std::cout << "    model" << std::endl;
    std::cout << "    approx" << std::endl;
    std::cout << "    sample_rate_rnn" << std::endl;
    std::cout << "    bad_model" << std::endl;
    std::cout << "    torch" << std::endl;
    for(auto& testConfig : tests)
        std::cout << "    " << testConfig.first << std::endl;
    std::cout << "    conv2d" << std::endl;
}

template <typename T>
int runTest(const TestConfig& test)
{
    std::cout << "TESTING " << test.name << " IMPLEMENTATION..." << std::endl;

    std::ifstream jsonStream(test.model_file, std::ifstream::binary);
    auto model = RTNeural::json_parser::parseJson<T>(jsonStream, true);
    model->reset();

    std::ifstream pythonX(test.x_data_file);
    auto xData = load_csv::loadFile<T>(pythonX);

    std::ifstream pythonY(test.y_data_file);
    const auto yRefData = load_csv::loadFile<T>(pythonY);

    std::vector<T> yData(xData.size(), (T)0);
    for(size_t n = 0; n < xData.size(); ++n)
    {
        T input[] = { xData[n] };
        yData[n] = model->forward(input);
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

int main(int argc, char* argv[])
{
#if RTNEURAL_USE_XSIMD
    std::cout << "XSIMD float register width: " << xsimd::simd_type<float>::size << std::endl;
    std::cout << "XSIMD double register width: " << xsimd::simd_type<double>::size << std::endl;
#endif

    if(argc != 2)
    {
        help();
        return 1;
    }

    std::string arg = argv[1];
    if(arg == "--help")
    {
        help();
        return 1;
    }

    if(arg == "all")
    {
        util_test();

        int result = 0;
        result |= model_test::model_test();
        result |= approximationTests();
        result |= sampleRateRNNTest();
        result |= conv2d_test();
        result |= torchGRUTest();
        result |= torchConv1DTest();
        result |= torchLSTMTest();

        for(auto& testConfig : tests)
        {
            result |= runTest<TestType>(testConfig.second);
            result |= templatedTests(testConfig.first);
        }

        return result;
    }

    if(arg == "conv2d")
    {
        return conv2d_test();
    }

    if(arg == "util")
    {
        util_test();
        return 0;
    }

    if(arg == "model")
    {
        return model_test::model_test();
    }

    if(arg == "approx")
    {
        return approximationTests();
    }

    if(arg == "sample_rate_rnn")
    {
        return sampleRateRNNTest();
    }

    if(arg == "bad_model")
    {
        return badModelTest();
    }

    if(arg == "torch")
    {
        int result = 0;
        result |= torchGRUTest();
        result |= torchConv1DTest();
        result |= torchLSTMTest();
        return result;
    }

    if(arg == "conv2d_model")
    {
        return conv2d_test();
    }

    if(tests.find(arg) != tests.end())
    {
        int result = 0;
        result |= runTest<TestType>(tests.at(arg));
        result |= templatedTests(arg);
        return result;
    }

    std::cout << "Test: " << arg << " not found!" << std::endl;
    return 1;
}
