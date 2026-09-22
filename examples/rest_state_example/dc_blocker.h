#pragma once

#include <cmath>

// y[n] = x[n] + s[n-1], s[n] = R * y[n] - x[n]
struct DCBlocker
{
    void prepare(float cutoffHz, float sampleRate)
    {
        R = std::exp(-2.0f * 3.14159265f * cutoffHz / sampleRate);
    }

    float process(float x)
    {
        const float y = x + s;
        s = R * y - x;
        return y;
    }

    void reset(float initialState = 0.0f)
    {
        s = initialState;
    }

    float R = 0.0f;
    float s = 0.0f;
};
