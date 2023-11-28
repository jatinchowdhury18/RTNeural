#include <gmock/gmock.h>

#include "load_csv.hpp"
#include <RTNeural/RTNeural.h>

using namespace testing;

using TestType = double;

namespace
{
template <typename ModelType>
void processModel(ModelType& model, const std::vector<TestType>& xData, std::vector<TestType>& yData)
{
    model.reset();
    for(size_t n = 0; n < xData.size(); ++n)
    {
        TestType input alignas(RTNEURAL_DEFAULT_ALIGNMENT)[] = { xData[n] };
        yData[n] = model.forward(input);
    }
}

const std::string model_file = std::string { RTNEURAL_ROOT_DIR } + "models/full_model.json";
const std::string data_file = std::string { RTNEURAL_ROOT_DIR } + "test_data/dense_x_python.csv";

auto loadInputData()
{
    std::ifstream pythonX(data_file);
    return load_csv::loadFile<TestType>(pythonX);
}

auto loadDynamicModel()
{
    std::ifstream jsonStream(model_file, std::ifstream::binary);
    return RTNeural::json_parser::parseJson<TestType>(jsonStream, true);
}

auto loadTemplatedModel()
{
    auto modelT = RTNeural::ModelT<TestType, 1, 1,
        RTNeural::DenseT<TestType, 1, 8>,
        RTNeural::TanhActivationT<TestType, 8>,
        RTNeural::Conv1DT<TestType, 8, 4, 3, 2>,
        RTNeural::TanhActivationT<TestType, 4>,
        RTNeural::GRULayerT<TestType, 4, 8>,
        RTNeural::DenseT<TestType, 8, 1>> {};

    std::ifstream jsonStream(model_file, std::ifstream::binary);
    modelT.parseJson(jsonStream, true);
    return modelT;
}
}

TEST(TestModel, templateModelOutputMatchesDynamicModel)
{
    constexpr double threshold = 1.0e-12;

    auto xData = loadInputData();
    auto yRefData = std::vector<TestType>(xData.size(), TestType { 0 });
    auto yData = std::vector<TestType>(xData.size(), TestType { 0 });

    auto modelRef = loadDynamicModel();
    processModel(*modelRef.get(), xData, yRefData);

    auto modelT = loadTemplatedModel();
    processModel(modelT, xData, yData);

    EXPECT_THAT(yData, Pointwise(DoubleNear(threshold), yRefData));
}
