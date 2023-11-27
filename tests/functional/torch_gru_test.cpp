#include <gmock/gmock.h>

#include "load_csv.hpp"
#include <RTNeural/RTNeural.h>

namespace
{
template <typename T>
void testTorchGRUModel()
{
    using ModelType = RTNeural::ModelT<T, 1, 1, RTNeural::GRULayerT<T, 1, 8>, RTNeural::DenseT<T, 8, 1>>;

    const auto loadModel = [](std::ifstream& jsonStream, ModelType& model)
    {
        nlohmann::json modelJson;
        jsonStream >> modelJson;

        auto& gru = model.template get<0>();
        RTNeural::torch_helpers::loadGRU<T>(modelJson, "gru.", gru);

        auto& dense = model.template get<1>();
        RTNeural::torch_helpers::loadDense<T>(modelJson, "dense.", dense);
    };

    const auto model_file = std::string { RTNEURAL_ROOT_DIR } + "models/gru_torch.json";
    std::ifstream jsonStream(model_file, std::ifstream::binary);

    ModelType model;
    loadModel(jsonStream, model);
    model.reset();

    std::ifstream modelInputsFile { std::string { RTNEURAL_ROOT_DIR } + "test_data/gru_torch_x_python.csv" };
    const auto inputs = load_csv::loadFile<T>(modelInputsFile);
    std::vector<T> outputs {};
    outputs.resize(inputs.size(), {});

    for(size_t i = 0; i < inputs.size(); ++i)
    {
        outputs[i] = model.forward(&inputs[i]);
    }

    std::ifstream modelOutputsFile { std::string { RTNEURAL_ROOT_DIR } + "test_data/gru_torch_y_python.csv" };
    const auto expected_y = load_csv::loadFile<T>(modelOutputsFile);

    using namespace testing;
    EXPECT_THAT(outputs, Pointwise(DoubleNear(1e-6), expected_y));
}
}

TEST(TestTorchGRU, modelOutputMatchesPythonImplementationForFloats)
{
    testTorchGRUModel<float>();
}

TEST(TestTorchGRU, modelOutputMatchesPythonImplementationForDoubles)
{
    testTorchGRUModel<double>();
}
