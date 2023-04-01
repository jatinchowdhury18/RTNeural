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
    using Vec2d = std::vector<std::vector<float>>;
    auto transpose = [](const Vec2d& x) -> Vec2d
    {
        auto outer_size = x.size();
        auto inner_size = x[0].size();
        Vec2d y(inner_size, std::vector<float>(outer_size, 0.0f));

        for(size_t i = 0; i < outer_size; ++i)
        {
            for(size_t j = 0; j < inner_size; ++j)
                y[j][i] = x[i][j];
        }

        return std::move(y);
    };

    const auto swap_rz = [](Vec2d& vec2d, int gru_size)
    {
        for(auto& vec : vec2d)
            std::swap_ranges(vec.begin(), vec.begin() + gru_size, vec.begin() + gru_size);
    };

    nlohmann::json modelJson;
    jsonStream >> modelJson;

    auto& gru = model.get<0>();

    // For the kernel and recurrent weights, PyTorch stores the weights similar to the
    // Tensorflow format, but transposed, and with the "r" and "z" indexes swapped.

    const std::vector<std::vector<float>> gru_ih_weights = modelJson.at("gru.weight_ih_l0");
    auto wVals = transpose(gru_ih_weights);
    swap_rz(wVals, gru.out_size);
    gru.setWVals(wVals);

    const std::vector<std::vector<float>> gru_hh_weights = modelJson.at("gru.weight_hh_l0");
    auto uVals = transpose(gru_hh_weights);
    swap_rz(uVals, gru.out_size);
    gru.setUVals(uVals);

    // PyTorch stores the GRU bias pretty much the same as TensorFlow as well,
    // just in two separate vectors. And again, we need to swap the "r" and "z" parts.

    const std::vector<float> gru_ih_bias = modelJson.at("gru.bias_ih_l0");
    const std::vector<float> gru_hh_bias = modelJson.at("gru.bias_hh_l0");
    std::vector<std::vector<float>> gru_bias { gru_ih_bias, gru_hh_bias };
    swap_rz(gru_bias, gru.out_size);
    gru.setBVals(gru_bias);

    auto& dense = model.get<1>();
    const std::vector<std::vector<float>> dense_weights = modelJson.at("dense.weight");
    dense.setWeights(dense_weights);
    const std::vector<float> dense_bias = modelJson.at("dense.bias");
    dense.setBias(dense_bias.data());
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
