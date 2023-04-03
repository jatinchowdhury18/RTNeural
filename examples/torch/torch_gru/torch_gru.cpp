#include "RTNeural/RTNeural.h"
#include "tests/load_csv.hpp"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

std::string getRootDir(fs::path path)
{
    while((--path.end())->string() != "RTNeural")
        path = path.parent_path();
    return path.string();
}

std::string getModelFile(fs::path path)
{
    path = getRootDir(path);
    path.append("models/gru_torch.json");

    return path.string();
}

std::string getInputFile(fs::path path)
{
    path = getRootDir(path);
    path.append("test_data/gru_torch_x_python.csv");
    return path.string();
}

std::string getOutputFile(fs::path path)
{
    path = getRootDir(path);
    path.append("test_data/gru_torch_y_python.csv");
    return path.string();
}

using ModelType = RTNeural::ModelT<float, 1, 1, RTNeural::GRULayerT<float, 1, 8>, RTNeural::DenseT<float, 8, 1>>;

void loadModel(std::ifstream& jsonStream, ModelType& model)
{
    nlohmann::json modelJson;
    jsonStream >> modelJson;

    auto& gru = model.get<0>();
    RTNeural::torch_helpers::loadGRU<float> (modelJson, "gru.", gru);

    auto& dense = model.get<1>();
    RTNeural::torch_helpers::loadDense<float> (modelJson, "dense.", dense);
}

int main([[maybe_unused]] int argc, char* argv[])
{
    std::cout << "Running \"torch conv1d\" example..." << std::endl;

    auto executablePath = fs::weakly_canonical(fs::path(argv[0]));
    auto modelFilePath = getModelFile(executablePath);

    std::cout << "Loading model from path: " << modelFilePath << std::endl;
    std::ifstream jsonStream(modelFilePath, std::ifstream::binary);

    ModelType model;
    loadModel(jsonStream, model);
    model.reset();

    std::ifstream modelInputsFile { getInputFile(executablePath) };
    const std::vector<float> inputs = load_csv::loadFile<float>(modelInputsFile);
    std::vector<float> outputs {};
    outputs.resize(inputs.size(), {});

    for(size_t i = 0; i < inputs.size(); ++i)
    {
        outputs[i] = model.forward(&inputs[i]);
    }

    std::ifstream modelOutputsFile { getOutputFile(executablePath) };
    const std::vector<float> expected_y = load_csv::loadFile<float>(modelOutputsFile);
    for(size_t i = 100; i < 105; ++i)
    {
        std::cout << expected_y[i] << std::endl;
        std::cout << outputs[i] << std::endl;
    }

    return 0;
}
