#pragma once

#include "load_csv.hpp"
#include <RTNeural.h>

void processModelNonT(RTNeural::Model<TestType>& model, const std::vector<TestType>& xData, std::vector<TestType>& yData, int num_frames, int num_features_in, int num_features_out)
{
    model.reset();

    TestType input alignas(RTNEURAL_DEFAULT_ALIGNMENT)[num_features_in];

    for(size_t n = 0; n < num_frames; ++n)
    {
        std::copy(xData.data() + n * num_features_in, xData.data() + (n + 1) * num_features_in, input);
        model.forward(input);
        std::copy(model.getOutputs(), model.getOutputs() + num_features_out, yData.data() + n * num_features_out);
    }
}

template <int numFeaturesIn, typename ModelType>
void processModelT(ModelType& model, const std::vector<TestType>& xData, std::vector<TestType>& yData, int num_frames, int num_features_out)
{
    model.reset();

    TestType input alignas(RTNEURAL_DEFAULT_ALIGNMENT)[numFeaturesIn];

    for(size_t n = 0; n < num_frames; n++)
    {
        std::copy(xData.begin() + n * numFeaturesIn, xData.begin() + (n + 1) * numFeaturesIn, input);
        auto input_mat = Eigen::Map<Eigen::Matrix<TestType, numFeaturesIn, 1>, RTNEURAL_DEFAULT_ALIGNMENT>(input);
        model.forward(input);
        std::copy(model.getOutputs(), model.getOutputs() + num_features_out, yData.data() + n * num_features_out);
    }
}

int conv2d_test()
{
    std::cout << "TESTING CONV2D MODEL..." << std::endl;

    const std::string model_file = "models/conv2d.json";
    const std::string data_file = "test_data/conv2d_x_python.csv";
    const std::string data_file_y = "test_data/conv2d_y_python.csv";

    constexpr double threshold = 1.0e-5;

    std::ifstream pythonX(data_file);
    auto xData = load_csv::loadFile<TestType>(pythonX);

    std::ifstream pythonY(data_file_y);
    auto yDataPython = load_csv::loadFile<TestType>(pythonY);

    int num_features_in;
    int num_frames;
    int num_features_out;

    // non-templated model
    std::vector<TestType> yData;
    {
        std::cout << "Loading non-templated model" << std::endl;
        std::ifstream jsonStream(model_file, std::ifstream::binary);
        auto modelRef = RTNeural::json_parser::parseJson<TestType>(jsonStream, true);

        num_features_in = modelRef->getInSize();
        num_frames = static_cast<int>(xData.size()) / num_features_in;
        num_features_out = modelRef->getOutSize();

        yData.resize(num_frames * num_features_out, (TestType)0);
        processModelNonT(*modelRef, xData, yData, num_frames, num_features_in, num_features_out);
    }

    size_t nErrs = 0;
    TestType max_error = (TestType)0;

    size_t shift = yData.size() - yDataPython.size();

    // Check for non templated
    for(size_t n = 0; n < yDataPython.size(); ++n)
    {
        auto err = std::abs(yDataPython.at(n) - yData.at(n + shift));
        if(err > threshold)
        {
            max_error = std::max(err, max_error);
            nErrs++;
        }
    }

    if(nErrs > 0)
    {
        std::cout << "FAIL NON TEMPLATED: " << nErrs << " errors!" << std::endl;
        std::cout << "Maximum error: " << max_error << std::endl;
        return 1;
    }

    std::cout << "SUCCESS NON TEMPLATED!" << std::endl
              << std::endl;

#if MODELT_AVAILABLE
    // templated model
    std::vector<TestType> yDataT(num_frames * num_features_out, (TestType)0);
    {
        std::cout << "Loading templated model" << std::endl;
        RTNeural::ModelT<TestType, 50, 10,
            RTNeural::Conv2DT<TestType, 1, 8, 50, 5, 3, 1, 3>,
            RTNeural::ReLuActivationT<TestType, 16 * 8>,
            RTNeural::Conv2DT<TestType, 8, 1, 16, 5, 7, 5, 1>,
            RTNeural::ReLuActivationT<TestType, 10>>
            modelT;
        //        RTNeural::ModelT<TestType, 1, 1, RTNeural::Conv2DT<TestType, 1, 1, 1, 1, 1, 1, 1>> modelT;

        std::ifstream jsonStream(model_file, std::ifstream::binary);
        modelT.parseJson(jsonStream, true);
        processModelT<50>(modelT, xData, yDataT, num_frames, num_features_out);
    }

    for(size_t n = 0; n < yDataPython.size(); ++n)
    {
        auto err = std::abs(yDataPython.at(n) - yDataT.at(n + shift));
        if(err > threshold)
        {
            max_error = std::max(err, max_error);
            nErrs++;
        }
    }

    if(nErrs > 0)
    {
        std::cout << "FAIL TEMPLATED: " << nErrs << " errors!" << std::endl;
        std::cout << "Maximum error: " << max_error << std::endl;
        return 1;
    }

    std::cout << "SUCCESS TEMPLATED!" << std::endl;

#endif

    return 0;
}
