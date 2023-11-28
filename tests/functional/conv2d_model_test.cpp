#include <gmock/gmock.h>

#include "load_csv.hpp"
#include <RTNeural/RTNeural.h>

namespace
{
using TestType = double;
static constexpr int num_features_in = 23;

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

const auto model_file = std::string { RTNEURAL_ROOT_DIR } + "models/conv2d.json";
const auto data_file = std::string { RTNEURAL_ROOT_DIR } + "test_data/conv2d_x_python.csv";
const auto data_file_y = std::string { RTNEURAL_ROOT_DIR } + "test_data/conv2d_y_python.csv";

auto loadPythonInputs()
{
    std::ifstream pythonX(data_file);
    return load_csv::loadFile<TestType>(pythonX);
}

auto loadPythonOutputs()
{
    std::ifstream pythonY(data_file_y);
    return load_csv::loadFile<TestType>(pythonY);
}

auto loadNonTemplatedModel()
{
    std::ifstream jsonStream(model_file, std::ifstream::binary);
    return RTNeural::json_parser::parseJson<TestType>(jsonStream, true);
}

auto loadTemplatedModel()
{
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
    return modelT;
}

template <typename ModelType>
void testModelOutputMatchesPythonImplementation(
    ModelType& model,
    int model_receptive_field,
    int tensorflow_pad_left,
    int num_features_out)
{
    constexpr double threshold = 1.0e-6;

    auto xData = loadPythonInputs();
    auto yDataPython = loadPythonOutputs();

    const auto num_frames = static_cast<int>(xData.size()) / num_features_in;

    auto yData = std::vector<TestType>(num_frames * num_features_out, TestType { 0 });
    processModel<num_features_in>(model, xData, yData, num_frames, num_features_out);

    // Evaluate only on valid range
    const auto start_frame_python = tensorflow_pad_left;
    const auto start_frame_rtneural = model_receptive_field - 1;
    const auto num_valid_frames = num_frames - start_frame_rtneural;

    auto y_index_getter = [](auto start_frame, auto num_features_out)
    {
        return [=](auto frame_index, auto feature_index)
        {
            return (start_frame + frame_index) * num_features_out + feature_index;
        };
    };

    auto y_index_python = y_index_getter(start_frame_python, num_features_out);
    auto y_index_rtneural = y_index_getter(start_frame_rtneural, num_features_out);

    for(size_t n_f = 0; n_f < num_valid_frames; ++n_f)
    {
        for(size_t i = 0; i < num_features_out; ++i)
        {
            EXPECT_NEAR(
                yDataPython.at(y_index_python(n_f, i)),
                yData.at(y_index_rtneural(n_f, i)),
                threshold);
        }
    }
}
}

TEST(TestConv2D, nonTemplatedModelOutputMatchesPythonImplementation)
{
#if RTNEURAL_AVX_ENABLED
    GTEST_SKIP() << "SKIPPING CONV2D MODEL TEST w/ AVX ENABLED...";
#else
    auto non_templated_model = loadNonTemplatedModel();
    ASSERT_THAT(non_templated_model, testing::Ne(nullptr));
    testModelOutputMatchesPythonImplementation(
        *non_templated_model,
        computeReceptiveField(*non_templated_model),
        computeTotalPaddedLeftFramesTensorflow(*non_templated_model),
        non_templated_model->getOutSize());
#endif
}

TEST(TestConv2D, templatedModelOutputMatchesPythonImplementation)
{
#if RTNEURAL_AVX_ENABLED
    GTEST_SKIP() << "SKIPPING CONV2D MODEL TEST w/ AVX ENABLED...";
#else
    auto non_templated_model = loadNonTemplatedModel();
    ASSERT_THAT(non_templated_model, testing::Ne(nullptr));

    auto templated_model = loadTemplatedModel();
    testModelOutputMatchesPythonImplementation(
        templated_model,
        computeReceptiveField(*non_templated_model),
        computeTotalPaddedLeftFramesTensorflow(*non_templated_model),
        non_templated_model->getOutSize());
#endif
}
