#include "rest_state.h"
#include "rest_state_model.h"

#include <iostream>

void printSilenceResponse(const char* title, RestStateModel& model, DCBlocker& dcBlocker)
{
    constexpr int numSamples = 10;
    const float silence = 0.0f;
    std::cout << title << std::endl;
    for(int n = 0; n < numSamples; ++n)
        std::cout << "  sample " << n << ": " << dcBlocker.process(model.forward(&silence)) << std::endl;
}

int main()
{
    RestStateModel model;
    loadRestStateModel(model);

    if((int)restState.size() != model.getStateSize())
    {
        std::cerr << "rest_state.h does not match the model" << std::endl;
        return 1;
    }

    DCBlocker dcBlocker;
    dcBlocker.prepare(dcBlockerCutoffHz, restStateSampleRate);

    model.reset();
    dcBlocker.reset();
    printSilenceResponse("Zero-state model and DC blocker, feeding silence:", model, dcBlocker);

    model.reset(restState.data());
    dcBlocker.reset(dcBlockerRestState);
    printSilenceResponse("Rest-state model and DC blocker, feeding silence:", model, dcBlocker);

    return 0;
}
