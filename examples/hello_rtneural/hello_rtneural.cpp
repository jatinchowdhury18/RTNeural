#include <iostream>
#include <filesystem>
#include <RTNeural/RTNeural.h>

namespace fs = std::filesystem;

std::string getModelFile (fs::path path)
{
    // get path of RTNeural root directory
    while((--path.end())->string() != "RTNeural")
        path = path.parent_path();

    // get path of model file
    path.append("examples/hello_rtneural/test_net.json");
    
    return path.string();
}

int main(int argc, char* argv[])
{
    std::cout << "Running \"hello rtneural\" example..." << std::endl;

    // get path of executable
    auto executablePath = fs::weakly_canonical(fs::path(argv[0]));
    auto modelFilePath = getModelFile(executablePath);

    std::cout << "Loading model from path: " << modelFilePath << std::endl;
    std::ifstream jsonStream(modelFilePath, std::ifstream::binary);
    auto model = RTNeural::json_parser::parseJson<float>(jsonStream, true);

    float testInput[1] = { 5.0f };
    float testOutput = model->forward (testInput);
    std::cout << "Test output: " << testOutput << std::endl;

    return 0;
}
