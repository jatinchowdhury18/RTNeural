#pragma once

#include "load_csv.hpp"
#include "util_tests.hpp"
#include <RTNeural.h>

/**
 * Computes the receptive field of the model (always 1 if there's no Conv2D layer in the model)
 * Also corresponds to the number of run after a reset before getting a valid/meaningful output
 */
int computeReceptiveField(const RTNeural::Model<TestType>& model)
{
    int receptive_field = 1;

    for(auto* l : model.layers)
    {
        if(l->getName() == "conv2d")
        {
            auto conv = dynamic_cast<RTNeural::Conv2D<TestType>*>(l);
            receptive_field += conv->receptive_field - 1;
        }
    }
    return receptive_field;
}

int computeTotalPaddedLeftFramesTensorflow(const RTNeural::Model<TestType>& model)
{
    int total_pad_left = 0;

    for(auto* l : model.layers)
    {
        if(l->getName() == "conv2d")
        {
            // Following tensorflow padding documentation: https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
            // And using the fact that in the time dimension, only stride = 1 is supported:

            auto conv = dynamic_cast<RTNeural::Conv2D<TestType>*>(l);

            if(!conv->valid_pad)
                total_pad_left += (conv->receptive_field - 1) / 2;
        }
    }

    return total_pad_left;
}

template <int numFeaturesIn, typename ModelType>
void processModel(ModelType& model, const std::vector<TestType>& xData, std::vector<TestType>& yData, int num_frames, int num_features_out)
{
    model.reset();

    TestType input alignas(RTNEURAL_DEFAULT_ALIGNMENT)[numFeaturesIn];

    for(size_t n = 0; n < num_frames; n++)
    {
        std::copy(xData.begin() + n * numFeaturesIn, xData.begin() + (n + 1) * numFeaturesIn, input);
        model.forward(input);
        std::copy(model.getOutputs(), model.getOutputs() + num_features_out, yData.data() + n * num_features_out);
    }
}

int conv2d_test()
{
#if RTNEURAL_AVX_ENABLED
    std::cout << "SKIPPING CONV2D MODEL TEST w/ AVX ENABLED..." << std::endl;
#else
    std::cout << "TESTING CONV2D MODEL..." << std::endl;

    const std::string model_file = "models/conv2d.json";
    const std::string data_file = "test_data/conv2d_x_python.csv";
    const std::string data_file_y = "test_data/conv2d_y_python.csv";

    constexpr double threshold = 1.0e-6;

    std::ifstream pythonX(data_file);
    auto xData = load_csv::loadFile<TestType>(pythonX);

    std::ifstream pythonY(data_file_y);
    auto yDataPython = load_csv::loadFile<TestType>(pythonY);

    static constexpr int num_features_in = 23;
    int num_frames;
    int num_features_out;

    int model_receptive_field;
    int tensorflow_pad_left;

    // non-templated model
    std::vector<TestType> yData;
    {
        std::cout << "Loading non-templated model" << std::endl;
        std::ifstream jsonStream(model_file, std::ifstream::binary);
        auto modelRef = RTNeural::json_parser::parseJson<TestType>(jsonStream, true);

        if(!modelRef)
        {
            std::cout << "INVALID CONV2D MODEL..." << std::endl;
            return 1;
        }

        model_receptive_field = computeReceptiveField(*modelRef);
        tensorflow_pad_left = computeTotalPaddedLeftFramesTensorflow(*modelRef);

        num_frames = static_cast<int>(xData.size()) / num_features_in;
        num_features_out = modelRef->getOutSize();

        yData.resize(num_frames * num_features_out, (TestType)0);
        processModel<num_features_in>(*modelRef, xData, yData, num_frames, num_features_out);
    }

    size_t nErrs = 0;
    auto max_error = (TestType)0;

    // Evaluate only on valid range
    size_t start_frame_python = tensorflow_pad_left;
    size_t start_frame_rtneural = model_receptive_field - 1;
    size_t num_valid_frames = num_frames - start_frame_rtneural;

    // Check for non templated
    for(size_t n_f = 0; n_f < num_valid_frames; ++n_f)
    {
        for(size_t i = 0; i < num_features_out; ++i)
        {
            auto err = std::abs(yDataPython.at((start_frame_python + n_f) * num_features_out + i) - yData.at((start_frame_rtneural + n_f) * num_features_out + i));
            if(err > threshold)
            {
                max_error = std::max(err, max_error);
                nErrs++;
            }
        }
    }

    if(nErrs > 0)
    {
        std::cout << "FAIL NON TEMPLATED: " << nErrs << " errors over " + std::to_string(num_valid_frames * num_features_out) + " values!" << std::endl;
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

        RTNeural::ModelT2D<TestType, 1, num_features_in, 1, 8,
            RTNeural::Conv2DT<TestType, 1, 2, num_features_in, 5, 5, 2, 1, true>,
            RTNeural::BatchNorm2DT<TestType, 2, 19, false>,
            RTNeural::ReLuActivationT<TestType, 2 * 19>,
            RTNeural::Conv2DT<TestType, 2, 3, 19, 4, 3, 1, 2, false>,
            RTNeural::BatchNorm2DT<TestType, 3, 10, true>,
            RTNeural::Conv2DT<TestType, 3, 1, 10, 2, 3, 3, 1, true>>
            modelT;

        std::ifstream jsonStream(model_file, std::ifstream::binary);
        modelT.parseJson(jsonStream, true);
        processModel<num_features_in>(modelT, xData, yDataT, num_frames, num_features_out);
    }

    // Check for non templated
    for(size_t n_f = 0; n_f < num_valid_frames; ++n_f)
    {
        for(size_t i = 0; i < num_features_out; ++i)
        {
            auto err = std::abs(yDataPython.at((start_frame_python + n_f) * num_features_out + i) - yDataT.at((start_frame_rtneural + n_f) * num_features_out + i));
            if(err > threshold)
            {
                max_error = std::max(err, max_error);
                nErrs++;
            }
        }
    }

    if(nErrs > 0)
    {
        std::cout << "FAIL TEMPLATED: " << nErrs << " errors!" << std::endl;
        std::cout << "Maximum error: " << max_error << std::endl;
        return 1;
    }

    std::cout << "SUCCESS TEMPLATED!" << std::endl;

#endif // MODELT_AVAILABLE
#endif // ! RTNEURAL_USE_AVX
    return 0;
}
