#pragma once

#include <map>
#include <string>

struct TestConfig
{
    std::string name;
    std::string model_file;
    std::string x_data_file;
    std::string y_data_file;
    const double threshold;
};

static std::map<std::string, TestConfig> tests {
    { "conv1d",
        TestConfig { "CONV1D", "models/conv.json", "test_data/conv_x_python.csv",
            "test_data/conv_y_python.csv", 1.0e-6 } },
    { "dense",
        TestConfig { "DENSE", "models/dense.json", "test_data/dense_x_python.csv",
            "test_data/dense_y_python.csv", 2.0e-8 } },
    { "gru", TestConfig { "GRU", "models/gru.json", "test_data/gru_x_python.csv", "test_data/gru_y_python.csv", 5.0e-6 } },
    { "lstm",
        TestConfig { "LSTM", "models/lstm.json", "test_data/lstm_x_python.csv",
            "test_data/lstm_y_python.csv", 1.0e-6 } },
};
