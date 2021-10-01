#include <iostream>
#include <RTNeural/RTNeural.h>

int main(int argc, char* argv[])
{
    std::cout << "Running \"rtneural dynamic model\" example..." << std::endl;

    if (argc != 2)
    {
        std::cout << "rtneural_dynamic_model needs to be called with exactly 1 argument!" << std::endl;
        return 1;
    }

    std::cout << "Loading model from path: " << argv[1] << std::endl;
    std::ifstream jsonStream(argv[1], std::ifstream::binary);
    auto model = RTNeural::json_parser::parseJson<float>(jsonStream, true);

    auto inputSize = model->layers[0]->in_size;
    std::vector<float> testInput (inputSize, 5.0f);

    float testOutput = model->forward (testInput.data());
    std::cout << "Test output: " << testOutput << std::endl;

    return 0;
}
