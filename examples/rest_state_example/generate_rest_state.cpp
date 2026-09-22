#include "rest_state_model.h"

#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

template <typename Values>
void writeArray(std::ofstream& header, const char* name, const Values& values)
{
    header << "constexpr std::array " << name << " {\n";
    for(const auto value : values)
        header << "    " << std::showpoint << value << "f,\n";
    header << "};\n\n";
}

int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        std::cerr << "Usage: generate_rest_state <output_header>" << std::endl;
        return 1;
    }

    RestStateModel model;
    loadRestStateModel(model);

    constexpr int silenceSamples = (int)(2.0f * restStateSampleRate);
    const float silence = 0.0f;
    model.reset();
    for(int n = 0; n < silenceSamples; ++n)
        model.forward(&silence);

    std::vector<float> restState;
    model.getState(restState);
    const float dcBlockerRestState = -model.getOutputs()[0];

    std::ofstream header(argv[1]);
    header << std::setprecision(std::numeric_limits<float>::max_digits10);
    header << "#pragma once\n\n#include <array>\n\n";
    writeArray(header, "restState", restState);
    header << "constexpr float dcBlockerRestState = " << std::showpoint << dcBlockerRestState << "f;\n";
    return 0;
}
