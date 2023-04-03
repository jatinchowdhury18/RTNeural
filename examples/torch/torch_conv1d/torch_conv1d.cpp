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
    nlohmann::json modelJson;
    jsonStream >> modelJson;

    RTNeural::ModelT<float, 1, 12, RTNeural::Conv1DT<float, 1, 12, 5, 1>> model;
    RTNeural::torch_helpers::loadConv1D<float> (modelJson, "", model.get<0>());
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
