#pragma once

#include "dc_blocker.h"

#include <RTNeural/RTNeural.h>
#include <fstream>

constexpr float restStateSampleRate = 48000.0f;
constexpr float dcBlockerCutoffHz = 20.0f;

using RestStateModel = RTNeural::ModelT<float, 1, 1,
    RTNeural::DenseT<float, 1, 8>,
    RTNeural::TanhActivationT<float, 8>,
    RTNeural::Conv1DT<float, 8, 4, 3, 2>,
    RTNeural::TanhActivationT<float, 4>,
    RTNeural::GRULayerT<float, 4, 8>,
    RTNeural::DenseT<float, 8, 1>>;

inline void loadRestStateModel(RestStateModel& model)
{
    std::ifstream jsonStream(RTNEURAL_ROOT_DIR "models/full_model.json", std::ifstream::binary);
    model.parseJson(jsonStream);
}
