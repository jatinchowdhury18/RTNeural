#pragma once

#include <RTNeural/RTNeural.h>

/** Static implementation of a custom gated activation layer. */
template <typename T, int in_sizet>
class GatedActivation
{
public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = in_size / 2;

    GatedActivation() = default;

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "gated_activation"; }

    /**
     * We're going to tell RTNeural to NOT treat this as an activation
     * layer, since RTNeural expects activation layers to have the same
     * inout/output sizes.
     */
    constexpr bool isActivation() const noexcept { return false; }

    void reset() { }

    /** Performs forward propagation for gated activation. */
    inline void forward(const T (&ins)[in_size]) noexcept
    {
        const auto sigmoid = [] (T x)
        {
            return (T) 1 / ((T) 1 + std::exp (-x));
        };

        for(int i = 0; i < out_size; ++i)
            outs[i] = std::tanh(ins[i]) * sigmoid(ins[i + out_size]);
    }

    T outs alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];
};
