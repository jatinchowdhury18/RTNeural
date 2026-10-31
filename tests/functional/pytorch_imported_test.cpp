#include <gmock/gmock.h>

#include "load_csv.hpp"
#include <RTNeural/RTNeural.h>

// Automated-GuitarAmpModelling and similar frameworks train recurrent models with
// PyTorch, and save the resulting weights as a json representation of the PyTorch
// state_dict. These tests check the two ways of getting such a model into RTNeural:
//
//  - converting the file into RTNeural's own json format (see python/modelToRTNeural.py)
//    and loading it with json_parser;
//  - copying the weights into a templated model by hand.
//
// Both are checked against the output of the original PyTorch model, generated
// in double precision (see python/pytorch_rnn_reference.py). Note that the
// reference is the output of the network itself: the "skip" field of the model
// file describes an input skip connection applied by the host around the
// network, which is not part of the model that RTNeural runs.

namespace
{
using TestType = double;
constexpr double error_thresh = 1.0e-12;

using Vec2d = std::vector<std::vector<TestType>>;

Vec2d transpose(const Vec2d& x)
{
    Vec2d y(x[0].size(), std::vector<TestType>(x.size(), (TestType)0));
    for(size_t i = 0; i < x.size(); ++i)
        for(size_t j = 0; j < x[i].size(); ++j)
            y[j][i] = x[i][j];
    return y;
}

template <typename ModelType>
std::vector<TestType> processModel(ModelType& model, const std::vector<TestType>& xData)
{
    std::vector<TestType> yData(xData.size(), (TestType)0);
    model.reset();
    for(size_t n = 0; n < xData.size(); ++n)
    {
        TestType input alignas(RTNEURAL_DEFAULT_ALIGNMENT)[] = { xData[n] };
        yData[n] = model.forward(input);
    }
    return yData;
}

std::vector<TestType> loadTestSignal(const std::string& file)
{
    std::ifstream stream { std::string { RTNEURAL_ROOT_DIR } + "test_data/" + file };
    return load_csv::loadFile<TestType>(stream);
}

nlohmann::json loadModelJson(const std::string& file)
{
    std::ifstream stream { std::string { RTNEURAL_ROOT_DIR } + "models/" + file, std::ifstream::binary };
    nlohmann::json json;
    stream >> json;
    return json;
}

void expectMatchesReference(const std::vector<TestType>& actual, const std::vector<TestType>& expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    for(size_t n = 0; n < actual.size(); ++n)
        EXPECT_THAT(actual[n], testing::DoubleNear(expected[n], error_thresh));
}
}

TEST(TestPyTorchImported, convertedModelMatchesPyTorchOutput)
{
    const auto modelJson = loadModelJson("pytorch_imported.json");
    auto model = RTNeural::json_parser::parseJson<TestType>(modelJson);
    ASSERT_NE(model, nullptr);

    expectMatchesReference(processModel(*model, loadTestSignal("pytorch_x.csv")),
        loadTestSignal("pytorch_y.csv"));
}

TEST(TestPyTorchImported, manuallyLoadedWeightsMatchPyTorchOutput)
{
    const auto weightsJson = loadModelJson("pytorch.json");

    const auto unit_type = weightsJson.at("model_data").at("unit_type").get<std::string>();
    const auto hidden_size = weightsJson.at("model_data").at("hidden_size").get<int>();
    ASSERT_EQ(unit_type, "LSTM");
    ASSERT_EQ(hidden_size, 12);

    const Vec2d rec_weights_ih = weightsJson.at("state_dict").at("rec.weight_ih_l0");
    const Vec2d rec_weights_hh = weightsJson.at("state_dict").at("rec.weight_hh_l0");
    const std::vector<TestType> rec_bias_ih = weightsJson.at("state_dict").at("rec.bias_ih_l0");
    std::vector<TestType> rec_bias_hh = weightsJson.at("state_dict").at("rec.bias_hh_l0");
    const Vec2d dense_weights = weightsJson.at("state_dict").at("lin.weight");
    const std::vector<TestType> dense_bias = weightsJson.at("state_dict").at("lin.bias");

    RTNeural::ModelT<TestType, 1, 1,
        RTNeural::LSTMLayerT<TestType, 1, 12>,
        RTNeural::DenseT<TestType, 12, 1>>
        model;

    auto& lstm = model.get<0>();
    lstm.setWVals(transpose(rec_weights_ih));
    lstm.setUVals(transpose(rec_weights_hh));

    // PyTorch splits the recurrent bias into an input and a hidden part
    for(size_t i = 0; i < rec_bias_hh.size(); ++i)
        rec_bias_hh[i] += rec_bias_ih[i];
    lstm.setBVals(rec_bias_hh);

    auto& dense = model.get<1>();
    dense.setWeights(dense_weights);
    dense.setBias(dense_bias.data());

    expectMatchesReference(processModel(model, loadTestSignal("pytorch_x.csv")),
        loadTestSignal("pytorch_y.csv"));
}
