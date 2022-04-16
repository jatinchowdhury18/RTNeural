#include <iostream>
#include <filesystem>
#include <RTNeural/RTNeural.h>

namespace fs = std::filesystem;

// include the implementation of the custom layer here
#if RTNEURAL_USE_XSIMD
#include "GatedActivation_xsimd.h"
#elif RTNEURAL_USE_EIGEN
#include "GatedActivation_Eigen.h"
#else
#include "GatedActivation_STL.h"
#endif

std::string getModelFile (fs::path path)
{
    // get path of RTNeural root directory
    while((--path.end())->string() != "RTNeural")
        path = path.parent_path();

    // get path of model file
    path.append("examples/custom_layer_model/test_net.json");

    return path.string();
}

int main(int argc, char* argv[])
{
    std::cout << "Running \"custom layer model\" example..." << std::endl;

    // get path of executable
    auto executablePath = fs::weakly_canonical(fs::path(argv[0]));
    auto modelFilePath = getModelFile(executablePath);

    std::cout << "Loading model from path: " << modelFilePath << std::endl;
    std::ifstream jsonStream(modelFilePath, std::ifstream::binary);

    // define static model
    RTNeural::ModelT<float, 1, 1,
        RTNeural::DenseT<float, 1, 8>,
        RTNeural::TanhActivationT<float, 8>,
        RTNeural::DenseT<float, 8, 8>,
        GatedActivation<float, 8>,              // add the custom layer to the model here!
        RTNeural::DenseT<float, 4, 1>> model;

    model.parseJson (jsonStream, true, { "gated_activation"}); // add the layer type to the list of custom layer expected by the parser

    // if the custom layer has weights, they will need to be loaded by hand here:
    // model.get<3>().load_weights(...);

    float testInput[1] = { 5.0f };
    float testOutput = model.forward (testInput);
    std::cout << "Test output: " << testOutput << std::endl;

    return 0;
}
