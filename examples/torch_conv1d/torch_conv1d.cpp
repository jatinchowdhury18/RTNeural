#include "../../tests/load_csv.hpp"
#include <RTNeural/RTNeural.h>
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
    //    path.append("examples/rtneural_static_model/test_net.json");
    path.append("models/conv1d_torch.json");

    return path.string();
}

std::string getInputFile(fs::path path)
{
    path = getRootDir(path);
    path.append("test_data/conv1d_torch_x_python.csv");
    return path.string();
}

std::string getOutputFile(fs::path path)
{
    path = getRootDir(path);
    path.append("test_data/conv1d_torch_y_python.csv");
    return path.string();
}

template <typename T>
std::vector<std::vector<T>> loadFile2D(std::ifstream& stream)
{
    using Vec2d = std::vector<std::vector<float>>;
    auto transpose = [] (const Vec2d& x) -> Vec2d
    {
        auto outer_size = x.size();
        auto inner_size = x[0].size();
        Vec2d y (inner_size, std::vector<float> (outer_size, 0.0f));

        for (size_t i = 0; i < outer_size; ++i)
        {
            for (size_t j = 0; j < inner_size; ++j)
                y[j][i] = x[i][j];
        }

        return std::move (y);
    };

    std::vector<std::vector<T>> vec;

    std::string line;
    if(stream.is_open()) {
        while(std::getline(stream, line)) {
            std::vector<float> lineVec;
            std::string num;
            for (auto ch : line) {
                if (ch == ',') {
                    lineVec.push_back(static_cast<T>(std::stod(num)));
                    num.clear();
                    continue;
                }

                num.push_back(ch);
            }

            lineVec.push_back(static_cast<T>(std::stod(num)));
            vec.push_back(lineVec);
        }

        stream.close();
    }

    return transpose(vec);
}

using ConvLayer = RTNeural::Conv1DT<float, 1, 12, 5, 1>;

void reverseKernels(std::vector<std::vector<std::vector<float>>>& conv_weights)
{
    for (auto& channel_weights : conv_weights)
    {
        for (auto& kernel : channel_weights)
        {
            std::reverse(kernel.begin(), kernel.end());
        }
    }
}

void loadModel(std::ifstream& jsonStream, ConvLayer& conv)
{
    nlohmann::json modelJson;
    jsonStream >> modelJson;
    //    std::cout << "Weights: " << modelJson["weight"].dump() << std::endl;
    //    std::cout << "Bias: " << modelJson["bias"].dump() << std::endl;

    std::vector<std::vector<std::vector<float>>> conv_weights = modelJson.at("weight");
    reverseKernels(conv_weights);
    conv.setWeights(conv_weights);

    std::vector<float> conv_bias = modelJson.at("bias");
    conv.setBias(conv_bias);
}

template <typename Container>
void printArray(const Container& arr)
{
    for (auto x : arr)
        std::cout << x << ", ";
    std::cout << '\n';
//    std::cout << arr[0] << '\n';
}

int main([[maybe_unused]] int argc, char* argv[])
{
    std::cout << "Running \"torch conv1d\" example..." << std::endl;

    auto executablePath = fs::weakly_canonical(fs::path(argv[0]));
    auto modelFilePath = getModelFile(executablePath);

    std::cout << "Loading model from path: " << modelFilePath << std::endl;
    std::ifstream jsonStream(modelFilePath, std::ifstream::binary);

    RTNeural::ModelT<float, 1, 12, ConvLayer> model;
    loadModel(jsonStream, model.get<0>());
    model.reset();

    std::ifstream modelInputsFile { getInputFile(executablePath) };
    const std::vector<float> inputs = load_csv::loadFile<float>(modelInputsFile);
    std::vector<std::array<float, 12>> outputs {};
    outputs.resize(inputs.size(), {});

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        model.forward(&inputs[i]);
        std::copy(model.getOutputs(), model.getOutputs() + 12, outputs[i].begin());
    }

    std::ifstream modelOutputsFile { getOutputFile(executablePath) };
    auto expected_y = loadFile2D<float> (modelOutputsFile);
    for (size_t i = 100; i < 105; ++i)
    {
        printArray(expected_y[i]);
        printArray(outputs[i+4]);
    }

    return 0;
}
