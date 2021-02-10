#include "load_csv.hpp"
#include <RTNeural.h>
#include <iostream>
#include <string>
#include <unordered_map>

// @TODO: make tests for both float and double precision

struct TestConfig
{
    std::string name;
    std::string model_file;
    std::string x_data_file;
    std::string y_data_file;
    const double threshold;
};

static std::unordered_map<std::string, TestConfig> tests {
    { "dense",
        TestConfig { "DENSE", "models/dense.json", "test_data/dense_x_python.csv",
            "test_data/dense_y_python.csv", 2.0e-8 } },
    { "gru", TestConfig { "GRU", "models/gru.json", "test_data/gru_x_python.csv", "test_data/gru_y_python.csv", 5.0e-6 } },
    { "lstm",
        TestConfig { "LSTM", "models/lstm.json", "test_data/lstm_x_python.csv",
            "test_data/lstm_y_python.csv", 1.0e-3 } }
};

void help()
{
    std::cout << "RTNeural test suite:" << std::endl;
    std::cout << "Usage: rtneural_tests <test_type>" << std::endl;
    std::cout << std::endl;
    std::cout << "Available test types are:" << std::endl;

    std::cout << "    "
              << "all" << std::endl;
    for(auto& testConfig : tests)
        std::cout << "    " << testConfig.first << std::endl;
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
        int result = 0;
        for(auto& testConfig : tests)
            result |= runTest<double>(testConfig.second);

        return result;
    }

    if(tests.find(arg) != tests.end())
    {
        return runTest<double>(tests.at(arg));
    }

    std::cout << "Test: " << arg << " not found!" << std::endl;
    return 1;
}
