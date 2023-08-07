#include "RTNeural/RTNeural.h"
#include "tests/load_csv.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

std::string getFileFromRoot(fs::path exe_path, const std::string& path)
{
    // get path of RTNeural root directory
    while((--exe_path.end())->string() != "RTNeural")
        exe_path = exe_path.parent_path();

    // get path of model file
    exe_path.append(path);

    return exe_path.string();
}

int main(int /*argc*/, char* argv[])
{
    auto executablePath = fs::weakly_canonical(fs::path(argv[0]));
    std::ifstream modelInputsFile { getFileFromRoot(executablePath, "test_data/conv_stateless_x_python.csv") };
    std::vector<float> inputs = load_csv::loadFile<float>(modelInputsFile);
    std::cout << "Data with size =  " << inputs.size() << " are loaded" << std::endl;

#if ! RTNEURAL_USE_EIGEN
    float inputs_array[128] {};
    std::copy (inputs.begin(), inputs.end(), std::begin (inputs_array));
#endif

    std::ifstream modelOutputsFile { getFileFromRoot(executablePath, "test_data/conv_stateless_y_python.csv") };
    std::vector<float> referenceOutputs = load_csv::loadFile<float>(modelOutputsFile);

    RTNeural::Conv1DStatelessT<float, 1, 128, 12, 65, 1, true> conv1;
    RTNeural::PReLUActivationT<float, 64 * 12> PRelu1;
    RTNeural::BatchNorm2DT<float, 12, 64, true> bn1;
    RTNeural::Conv1DStatelessT<float, 12, 64, 8, 33, 1, true> conv2;
    RTNeural::PReLUActivationT<float, 32 * 8> PRelu2;
    RTNeural::BatchNorm2DT<float, 8, 32, true> bn2;
    RTNeural::Conv1DStatelessT<float, 8, 32, 4, 13, 1, true> conv3;
    RTNeural::PReLUActivationT<float, 20 * 4> PRelu3;
    RTNeural::BatchNorm2DT<float, 4, 20, true> bn3;
    RTNeural::Conv1DStatelessT<float, 4, 20, 1, 5, 1, true> conv4;
    RTNeural::TanhActivationT<float, 16> tanh {};

    auto modelFilePath = getFileFromRoot(executablePath, "models/conv_stateless.json");

    std::cout << "Loading model from path: " << modelFilePath << std::endl;
    std::ifstream jsonStream(modelFilePath, std::ifstream::binary);

    nlohmann::json modelJson;
    jsonStream >> modelJson;
    const auto layersJson = modelJson.at ("layers");

    conv1.setWeightsTransposed (layersJson.at (0).at ("weights").at (0).get<std::vector<std::vector<std::vector<float>>>>());

    RTNeural::json_parser::loadPReLU<float> (PRelu1, layersJson.at (1).at ("weights"));
    bn1.setEpsilon (layersJson.at (2).at ("epsilon"));
    RTNeural::json_parser::loadBatchNorm<float> (bn1, layersJson.at (2).at ("weights"), true);
    conv2.setWeightsTransposed (layersJson.at (3).at ("weights").at (0).get<std::vector<std::vector<std::vector<float>>>>());

    RTNeural::json_parser::loadPReLU<float> (PRelu2, layersJson.at (4).at ("weights"));
    bn2.setEpsilon (layersJson.at (5).at ("epsilon"));
    RTNeural::json_parser::loadBatchNorm<float> (bn2, layersJson.at (5).at ("weights"), true);
    conv3.setWeightsTransposed (layersJson.at (6).at ("weights").at (0).get<std::vector<std::vector<std::vector<float>>>>());

    RTNeural::json_parser::loadPReLU<float> (PRelu3, layersJson.at (7).at ("weights"));
    bn3.setEpsilon (layersJson.at (8).at ("epsilon"));
    RTNeural::json_parser::loadBatchNorm<float> (bn3, layersJson.at (8).at ("weights"), true);
    conv4.setWeightsTransposed (layersJson.at (9).at ("weights").at (0).get<std::vector<std::vector<std::vector<float>>>>());

    conv1.reset();
    PRelu1.reset();
    bn1.reset();
    conv2.reset();
    PRelu2.reset();
    bn2.reset();
    conv3.reset();
    PRelu3.reset();
    bn3.reset();
    conv4.reset();

    std::vector<float> testOutputs;
    testOutputs.resize(referenceOutputs.size(), 0.0f);

    namespace chrono = std::chrono;
    const auto start = chrono::high_resolution_clock::now();

#if RTNEURAL_USE_EIGEN
    conv1.forward (Eigen::Map<Eigen::Matrix<float, 1, 128>> { inputs.data() });

    PRelu1.forward (Eigen::Map<Eigen::Vector<float, 64 * 12>> { conv1.outs.data() });
    bn1.forward (Eigen::Map<Eigen::Vector<float, 64 * 12>> { PRelu1.outs.data() });
    conv2.forward (Eigen::Map<Eigen::Matrix<float, 12, 64>> { bn1.outs.data() });

    PRelu2.forward (Eigen::Map<Eigen::Vector<float, 32 * 8>> { conv2.outs.data() });
    bn2.forward (Eigen::Map<Eigen::Vector<float, 32 * 8>> { PRelu2.outs.data() });
    conv3.forward (Eigen::Map<Eigen::Matrix<float, 8, 32>> { bn2.outs.data() });

    PRelu3.forward (Eigen::Map<Eigen::Vector<float, 20 * 4>> { conv3.outs.data() });
    bn3.forward (Eigen::Map<Eigen::Vector<float, 20 * 4>> { PRelu3.outs.data() });
    conv4.forward (Eigen::Map<Eigen::Matrix<float, 4, 20>> { bn3.outs.data() });
    tanh.forward (Eigen::Map<Eigen::Vector<float, 16>> { conv4.outs.data() });
#else
    conv1.forward (inputs_array);

    PRelu1.forward (conv1.outs);
    bn1.forward (PRelu1.outs);
    conv2.forward (bn1.outs);

    PRelu2.forward (conv2.outs);
    bn2.forward (PRelu2.outs);
    conv3.forward (bn2.outs);

    PRelu3.forward (conv3.outs);
    bn3.forward (PRelu3.outs);
    conv4.forward (bn3.outs);
    tanh.forward (conv4.outs);
#endif

    const auto duration = chrono::high_resolution_clock::now() - start;
    std::cout << "Time taken by function: " << chrono::duration_cast<chrono::microseconds>(duration).count() << " microseconds" << std::endl;

    for(size_t i = 0; i < 16; ++i)
    {
        std::cout << referenceOutputs[i] << " | " << tanh.outs[i] << std::endl;
    }

    return 0;
}
