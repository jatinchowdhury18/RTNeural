#pragma once

#include "load_csv.hpp"
#include <RTNeural.h>

void processModelNonT(RTNeural::Model<TestType>& model, const std::vector<TestType>& xData, std::vector<TestType>& yData, int num_features_in, int num_features_out)
{
    model.reset();

    int num_frames = static_cast<int>(xData.size()) / num_features_in;
    TestType input alignas(RTNEURAL_DEFAULT_ALIGNMENT)[num_features_in];

    for(size_t n = 0; n < num_frames; ++n)
    {
        std::copy(xData.data() + n * num_features_in, xData.data() + (n + 1) * num_features_in, input);
        model.forward(input);
        std::copy(model.getOutputs(), model.getOutputs() + num_features_out, yData.data() + n * num_features_out);
    }
}

// template <int numFeaturesIn, typename ModelType>
// void processModelT(ModelType& model, const std::vector<TestType>& xData, std::vector<TestType>& yData)
//{
//     model.reset();
//     int num_frames = xData.size() / numFeaturesIn;
//
//     TestType input alignas(RTNEURAL_DEFAULT_ALIGNMENT)[numFeaturesIn];
//
//     for(size_t n = 0; n < num_frames; n++)
//     {
//         std::copy(xData.begin() + n * numFeaturesIn, xData.begin() + (n + 1) * numFeaturesIn, input);
//         auto input_mat = Eigen::Map<Eigen::Matrix<TestType, numFeaturesIn, 1>, RTNEURAL_DEFAULT_ALIGNMENT>(input);
//         model.forward(input);
//         std::copy(model.getOutputs(), model.getOutputs() + model.num_features_out, yData.data() + n * model.num_features_out);
//     }
// }

int conv2d_test()
{
    std::cout << "TESTING CONV2D MODEL..." << std::endl;

    const std::string model_file = "models/conv2d.json";
    const std::string data_file = "test_data/conv2d_x_python.csv";
    const std::string data_file_y = "test_data/conv2d_y_python.csv";

    constexpr double threshold = 1.0e-5;

    const int num_features_in = 5;
    const int num_features_out = 1;

    std::ifstream pythonX(data_file);
    auto xData = load_csv::loadFile<TestType>(pythonX);
    int num_frames = xData.size() / num_features_in;

    std::ifstream pythonY(data_file_y);
    auto yDataPython = load_csv::loadFile<TestType>(pythonY);

    // non-templated model
    std::vector<TestType> yRefData(num_frames * num_features_out, (TestType)0);
    {
        std::cout << "Loading non-templated model" << std::endl;
        std::ifstream jsonStream(model_file, std::ifstream::binary);
        auto modelRef = RTNeural::json_parser::parseJson<TestType>(jsonStream, true);
        processModelNonT(*modelRef, xData, yRefData, num_features_in, num_features_out);
    }

#if MODELT_AVAILABLE
    //    // templated model
    //    std::vector<TestType> yData(xData.size(), (TestType)0);
    //    {
    //        std::cout << "Loading templated model" << std::endl;
    //        RTNeural::ModelT<TestType, 49, 1,
    //            RTNeural::Conv2DT<TestType, 1, 4, 49, 3, 5, 1, 1>,
    //            RTNeural::BatchNorm1DT<TestType, 4 * 49, true>,
    //            RTNeural::ReLuActivationT<TestType, 4 * 49>,
    //            RTNeural::Conv2DT<TestType, 4, 8, 49, 5, 7, 2, 1>,
    //            RTNeural::BatchNorm1DT<TestType, 8 * 49, true>,
    //            RTNeural::ReLuActivationT<TestType, 8 * 49>,
    //            RTNeural::Conv2DT<TestType, 8, 1, 49, 5, 7, 1, 2>,
    //            RTNeural::BatchNorm1DT<TestType, 25, true>,
    //            RTNeural::ReLuActivationT<TestType, 25>>
    //            modelT;
    //        std::ifstream jsonStream(model_file, std::ifstream::binary);
    //        modelT.parseJson(jsonStream, true);
    //        processModelT<49>(modelT, xData, yData);
    //    }

    size_t nErrs = 0;
    TestType max_error = (TestType)0;

    size_t shift = yRefData.size() - yDataPython.size();

    for(size_t n = 0; n < yDataPython.size(); ++n)
    {
        auto err = std::abs(yDataPython.at(n) - yRefData.at(n +  shift));
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
